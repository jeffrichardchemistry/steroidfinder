#from numba.core import descriptors
import rdkit
import streamlit as st
import pandas as pd
import numpy as np
from SpectraFP import SpectraFP
try:
    from fastsimilarity import getOnematch
except:
    from .fastsimilarity import getOnematch
    
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from PIL import Image
import pickle
import os
ABSOLUT_PATH = os.path.dirname(os.path.realpath(__file__))



class BackEnd:
    def __init__(self):
        var = 0 
        self.df, self.df_fp1, self.df_fp2, self.df_fp3, self.df_fp4 = BackEnd.load_data(self) #Run code to get datasets as objects

        self.rf = None
        self.xgb = None
        self.mlp = None
        BackEnd.load_models(self)

    # Load datasets and store in memory cache
    @st.cache_data
    def load_data(_self):

        df_fp1 = pd.read_csv(f'{ABSOLUT_PATH}/data/df_fp1_all_EI.csv', converters={'m/z':eval})
        df_fp1.drop(['Unnamed: 0'], axis=1, inplace=True)

        df_fp2 = pd.read_csv(f'{ABSOLUT_PATH}/data/df_fp2_all_EI.csv', converters={'m/z':eval})
        df_fp2.drop(['Unnamed: 0'], axis=1, inplace=True)

        df_fp3 = pd.read_csv(f'{ABSOLUT_PATH}/data/df_fp3_all_EI.csv', converters={'m/z':eval})
        df_fp3.drop(['Unnamed: 0'], axis=1, inplace=True)

        df_fp4 = pd.read_csv(f'{ABSOLUT_PATH}/data/df_fp4_all_EI.csv', converters={'m/z':eval})
        df_fp4.drop(['Unnamed: 0'], axis=1, inplace=True)
        
        df = pd.read_csv(f'{ABSOLUT_PATH}/data/df_all_EI.csv', converters={'m/z':eval})
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        return df, df_fp1, df_fp2, df_fp3, df_fp4


    def load_models(self):
        rf   = f'{ABSOLUT_PATH}/Class_steroids/model_RFC_testo.sav'
        xgb  = f'{ABSOLUT_PATH}/Class_steroids/model_GBC_testo.sav'
        mlp  = f'{ABSOLUT_PATH}/Class_steroids/model_NN_testo.sav'
        

        self._rf   = pickle.load(open(rf, 'rb'))
        self._xgb  = pickle.load(open(xgb, 'rb'))
        # self._mlp  = pickle.load(open(mlp, 'rb'))

    # builder svg
    def __moltosvg(self, mol, molSize = (300,300), kekulize = True):
        mol = Chem.MolFromSmiles(mol)
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:','')

    # render svg    
    def render_svg(self, smiles):
        svg = BackEnd.__moltosvg(self, mol=smiles)
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        #st.write(html, unsafe_allow_html=True)
        return(html)

    ## match
    def match_FP(self, user_input, degree_freedom , df_fpx, threshold,  metric):
       
        lista = user_input.split(',')
        mz_peak_list = [float(x) for x in lista]
        mzs =  np.array(mz_peak_list)
        fpmz = SpectraFP(range_spectra=[14.0, 846.0, 0.1])
        fp_dfx = []
        fp_dfx = fpmz.genFP(mzs, correction=degree_freedom, spurious_variables=False)
        fp_dfx = np.array([fp_dfx])

        
        df_completo = pd.concat([self.df,df_fpx], axis=1)



        basex = df_completo.iloc[:,2:].values.astype('uint32')        
        var = getOnematch(base_train = basex, base_test = fp_dfx, complete_base = df_completo, similarity_metric=metric, alpha=1, beta=1, threshold=threshold)
        var = dict(sorted(var.items(), key=lambda x: x[1], reverse=True))
        return(var)

    
    # 4 cycle Filter
    def n4cycleFilter(self,var):
        mol = []
        smi = []
        steroids =[]
        steroids_index = []
        sim_ = []
        for smiles, sims_ in var.items():
        #for i in range(0 ,len(df['Smiles'])):
            mol.append(Chem.MolFromSmiles(smiles))
            sim_.append(sims_)
        for i,z in enumerate(mol):
            try:
                nali = Descriptors.NumAliphaticCarbocycles(z)
            except:
                pass
            if nali == 4:
                steroids_index.append(i)
                smi.append(Chem.MolToSmiles(z)) 
            
        return smi, sim_  


    # steroid mass matrix builder
    def frag_matrix_builder(self,mass,intencity_rel,exact_mass):
        ## separando m/z
        mz_temp = [int(float(x)) for x in mass]
        
        ## separando int.rel
        rel_temp = [int(float(x))/10 for x in intencity_rel]

        # massa exata
        exact_mz = exact_mass

        # buscando lista de m/z (artigo)
        mz4steroid_subtracao = [15,29,90,180,270,105,195,285,119,209,103,193,283,143,155,140,157,144]
        sinais_mz = [103,129,143,169,244,218,231]
        
        # aplicando subtracao da massa exata para gerar os possiveis fragmentos
        list_frags = []
        frags = []
        for m in range(0, len(mz4steroid_subtracao)):
            frags.append(exact_mz - mz4steroid_subtracao[m])
        list_frags += [frags]
        frags = []
        
        # colocando pesos nas intensidade relativa na possição correção
        weight_df = pd.DataFrame(np.zeros((1,len(mz4steroid_subtracao)), dtype = int))
        colunas = [str(x) for x in mz4steroid_subtracao]
        weight_df.columns = colunas
        #weight_df
        
        # encontrando os index dos fragmentos presentes na amostra
        for ind, elementos in enumerate(mz_temp):
            if elementos in list_frags[0]:
                index_frag = list_frags[0].index(elementos)
                if rel_temp[ind] < 25.0:
                    weight_df.iloc[0, index_frag] = 1
                elif rel_temp[ind] > 25.0 and rel_temp[ind] < 50:
                    weight_df.iloc[0, index_frag] = 2
                else:
                    weight_df.iloc[0, index_frag] = 3
            else:
                    pass
        
        # weight_df
        # encontrando sinais mz
        weight_df2 = pd.DataFrame(np.zeros((1,len(sinais_mz)), dtype = int))
        colunas = [str(x) for x in sinais_mz]
        weight_df2.columns = colunas
        for ind, elementos in enumerate(mz_temp):
            if elementos in sinais_mz:
                index_frag = sinais_mz.index(elementos)
                if rel_temp[ind] < 25.0:
                    weight_df2.iloc[0, index_frag] = 1
                elif rel_temp[ind] > 25.0 and rel_temp[ind] < 50:
                    weight_df2.iloc[0, index_frag] = 2
                else:
                    weight_df2.iloc[0, index_frag] = 3
            else:
                pass
        
        # encontrando peso da massa exata
        index_exact = mz_temp.index(exact_mz)
        intencity = rel_temp[index_exact]
        if intencity < 25.0:
            exact_weight = 1
        elif intencity > 25.0  and intencity < 50.0:
            exact_weight = 2
        else:
            exact_weight = 3
        

        # concatenando fragmentos e sinais m/z
        weight_df = weight_df.values.tolist()
        weight_df2 = weight_df2.values.tolist()
        list_final =list([exact_weight,*weight_df[0],*weight_df2[0]])
        
        
        return list_final
    
    # modelo de classifição de esteroides anabolicos e androgenicos
    def run_anabolic_model(self,descriptor_ms):
        
        result_rt = self._rf.predict([descriptor_ms])
        #result_xgb = self._xgb.predict([descriptor_ms])
        return result_rt



################################ GUI ##########################################################
class FrontEnd(BackEnd):
    def __init__(self):
        super().__init__()
        FrontEnd.main(self)


    def NavigationBar(self):
        st.sidebar.markdown('# Navegation:')
        nav = st.sidebar.radio('Go to:', ['Steroid Classifier', 'Similarity Analise' ,'Home'])
        
        st.sidebar.markdown('# Contribute')
        #st.sidebar.info('{}'.format(self.text2))
        
        return nav
       
   
    def main(self):
        nav = FrontEnd.NavigationBar(self)
        if nav == 'Similarity Analise':
            image = Image.open(f'{ABSOLUT_PATH}/image/SteroidFinder_logo.png')
            st.sidebar.image(image, width=250)

            user_input_mz = st.text_input("Input your m/z peak list ", '10.0,18.1,25.8,90.1')

            #columns
            col1, col2, col3 = st.columns(3)

            # bucket finger print m/z select box
            values = [1, 2, 3, 4]
            default_ix = values.index(4)
            Degree_freedom_sel = st.sidebar.selectbox('Degree freedom', values, index=default_ix)

            if Degree_freedom_sel==1:
                degree_freedom = 1 
                df_fpx = self.df_fp1
            if Degree_freedom_sel==2:
                degree_freedom = 2 
                df_fpx = self.df_fp2
            if Degree_freedom_sel==3:
                degree_freedom = 3 
                df_fpx = self.df_fp3
            if Degree_freedom_sel==4:
                degree_freedom = 4 
                df_fpx = self.df_fp4


            # bottom search similarity
            btn = col2.button('Serch Similarity') 

            btn_filter = col2.button('Cycle filter') 

            ## sidebar
            # slider threshold 
            number = st.sidebar.slider("Pick a threshold", 0.0, 1.0, 0.1)

            # select metric
            metrics = ['geometric','arithmetic','tanimoto','tversky']
            default_metric = metrics.index('tanimoto')
            metrics_sel = st.sidebar.selectbox('Metrics',metrics, index = default_metric)

            # Search Similarity and draw structure
            if btn:
                var = FrontEnd.match_FP(self,user_input=user_input_mz, degree_freedom = Degree_freedom_sel, df_fpx = df_fpx, threshold= number, metric = metrics_sel)
                for smiles, sims_ in var.items():
                    col2.write(FrontEnd.render_svg(self, smiles=smiles),unsafe_allow_html=True)
                    col2.markdown('# <center>{:.2f}%</center>'.format(sims_*100), unsafe_allow_html=True)

            # Search Similarity and draw structure using n4cycleFilter
            if btn_filter:
                var = FrontEnd.match_FP(self,user_input=user_input_mz, degree_freedom = Degree_freedom_sel, df_fpx = df_fpx, threshold= number, metric = metrics_sel)
                smi, sim_ = FrontEnd.n4cycleFilter(self,var)
                for smiles in range(0,len(smi)):
                    col2.write(FrontEnd.render_svg(self, smiles=smi[smiles]),unsafe_allow_html=True)
                    col2.markdown('# <center>{:.2f}%</center>'.format(sim_[smiles]*100), unsafe_allow_html=True)
        if nav == 'Home':
            st.title('Home test')
         
         # steroid predict AAS
        if nav == 'Steroid Classifier':
            image = Image.open(f'{ABSOLUT_PATH}/image/SteroidFinder_logo.png')
            st.sidebar.image(image, width=250)

            mass = list(st.text_input("Input your m/z peak list ", '10.0,18.1,25.8,90.1,422.0').split(','))
            intencity_rel = list(st.text_input("Input your intencity relative", '200,123,110,80,1000').split(','))
            exact_mass = float(st.text_input('Input your ion mass', '422'))
            col1, col2, col3 = st.columns(3)

            btn_ = col2.button('Build fragment matrix') 
            if btn_:
                var_ = FrontEnd.frag_matrix_builder(self,mass,intencity_rel,exact_mass)
                st.write(pd.DataFrame(var_))
                result = FrontEnd.run_anabolic_model(self,var_)
                if result == 1:
                    st.markdown("# This sample was classifier as doping")
                else:
                    st.markdown("# This sample was classifier not doping")

                

                

fe = FrontEnd()



