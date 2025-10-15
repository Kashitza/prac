:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="MhgYfsEFeS40" outputId="1fd455d5-09ee-4a06-cf7f-0b83d326e187"}
``` python
!pip install rdkit
```

::: {.output .stream .stdout}
    Collecting rdkit
      Downloading rdkit-2025.9.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (4.1 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from rdkit) (2.0.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from rdkit) (11.3.0)
    Downloading rdkit-2025.9.1-cp312-cp312-manylinux_2_28_x86_64.whl (36.2 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.2/36.2 MB 12.0 MB/s eta 0:00:00
:::
::::

::: {.cell .code id="9AZf9QisdjJV"}
``` python
  from rdkit import Chem
  from rdkit.Chem import AllChem
  from rdkit import RDConfig
  from rdkit.Chem.Draw import IPythonConsole
  from rdkit.Chem import Draw
  import numpy as np
  from IPython.display import display,Image
```
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":167}" id="9TAdzdy5dksi" outputId="608ff89c-01fb-4131-bd43-c246ce8f64af"}
``` python
  ibu=Chem.MolFromSmiles('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')
  AllChem.Compute2DCoords(ibu)
  display(ibu)
```

::: {.output .display_data}
![](a50cb99cdc311febb0f1543d35e73fa7c8bcb348.png)
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="PckQ4YrXeZ0Z" outputId="03f01786-5002-453c-ab2b-f969d6503e35"}
``` python
  import rdkit.Chem.Lipinski as Lipinksy
  print (Lipinksy.NumHDonors(ibu))
  print (Lipinksy.NumHAcceptors(ibu))
  print (Lipinksy.rdMolDescriptors.CalcExactMolWt(ibu))
  print (Lipinksy.rdMolDescriptors.CalcCrippenDescriptors(ibu)[0])
```

::: {.output .stream .stdout}
    1
    1
    206.130679816
    3.073200000000001
:::
::::

::::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":185}" id="uhpLiSehC6qG" outputId="1451b044-618a-47b9-bbd5-b3e5970bcbce"}
``` python
isopropyl_smarts = '[C](C)C'
acetylene_mol = Chem.MolFromSmiles('C#C')

rxn = Chem.ReplaceSubstructs(ibu, Chem.MolFromSmarts(isopropyl_smarts), acetylene_mol, replaceAll=False)
modified = rxn[0]
AllChem.Compute2DCoords(modified)

# Визуализация
Draw.MolToImage(modified)
display(modified)

# Для SMILES результата:
print(Chem.MolToSmiles(modified))
```

::: {.output .display_data}
![](ab8582e6ce8ff5402a8c2e87b068e55e33c80aea.png)
:::

::: {.output .stream .stdout}
    C#CCc1ccc(C(C)C(=O)O)cc1
:::
:::::

::: {.cell .code id="Rk0Fynjsh1Gb"}
``` python
import pandas as pd
```
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="VEl4y9_ZerYx" outputId="d51ef14b-7a3a-478e-d98d-312eeb150eff"}
``` python
smiles_df = pd.read_csv('/content/azides.csv')

# Фильтруем SMILES: длина меньше 30 и без '.'
smiles = []
for smiles_str in smiles_df['SMILES']:
    if isinstance(smiles_str, str) and '.' not in smiles_str:
        smiles.append(smiles_str)

print(smiles[:10],f'{len(smiles)}')
```

::: {.output .stream .stdout}
    ['CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)CO)N=[N+]=[N-]', 'N=[N+]=[N-]', 'CCOC(=O)C1=C2CN(C(=O)C3=C(N2C=N1)C=CC(=C3)N=[N+]=[N-])C', 'CC(C)NC1=NC(=NC(=N1)SC)N=[N+]=[N-]', '[N-]=[N+]=N[Pb]N=[N+]=[N-]', 'C1=CN(C(=O)N=C1N)[C@H]2[C@H]([C@@H]([C@](O2)(CO)N=[N+]=[N-])O)F', 'C1=CC=C(C=C1)CN=[N+]=[N-]', 'CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)N=[N+]=[N-]', 'C1=CC(=CC=C1[C@H]([C@@H](CO)NC(=O)CN=[N+]=[N-])O)[N+](=O)[O-]', 'CC1=CN(C(=O)N=C1N)[C@H]2C[C@@H]([C@H](O2)CO)N=[N+]=[N-]'] 370
:::
::::

::: {.cell .code id="w8rEHTnND3og"}
``` python
azide_smarts = '[$([NA]=[NA+]=[NA-])]'
```
:::

::::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":720}" id="f0fe8J1JEHzP" outputId="ef93db2a-425e-4d2d-fe2e-56448c546678"}
``` python
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors, Draw
from IPython.display import display

passed_mols = []
passed_smiles = []


split_df = []
mod_ibu_fragment = "(N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O)"

for smi in smiles[:1500]:  # или для pandas DataFrame: for i, row in smiles_df.iterrows():
    spl = smi.split("N=[N+]=[N-]")
    if len(spl) < 2:
        continue
    elif len(spl) > 2:
        spl = [spl[0], "N=[N+]=[N-]".join(spl[1:])]

    new_smi = ''.join([spl[0], mod_ibu_fragment, spl[1]])
    try:
        newmol = Chem.MolFromSmiles(new_smi)
        if newmol is None:
            continue
        donors = Lipinski.NumHDonors(newmol)
        acceptors = Lipinski.NumHAcceptors(newmol)
        mw = Descriptors.ExactMolWt(newmol)
        logp = Descriptors.MolLogP(newmol)
        if (donors <= 5 and acceptors <= 10 and mw < 500 and logp <= 5):
            split_df.append({
                # Для DataFrame можно добавить CID, если есть
                "source": smi,
                "result": new_smi
            })
    except Exception:
        pass

total_df = pd.DataFrame(split_df)
```

::: {.output .stream .stderr}
    [11:30:00] SMILES Parse Error: syntax error while parsing: (N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O)
    [11:30:00] SMILES Parse Error: check for mistakes around position 1:
    [11:30:00] (N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(
    [11:30:00] ^
    [11:30:00] SMILES Parse Error: Failed parsing SMILES '(N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O)' for input: '(N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O)'
    [11:30:00] SMILES Parse Error: syntax error while parsing: C[Si](C)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))N=[N+]=[N-]
    [11:30:00] SMILES Parse Error: check for mistakes around position 10:
    [11:30:00] C[Si](C)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%
    [11:30:00] ~~~~~~~~~^
    [11:30:00] SMILES Parse Error: Failed parsing SMILES 'C[Si](C)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))N=[N+]=[N-]' for input: 'C[Si](C)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))N=[N+]=[N-]'
    [11:30:00] SMILES Parse Error: syntax error while parsing: C1=CC=C(C=C1)OP(=O)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))OC2=CC=CC=C2
    [11:30:00] SMILES Parse Error: check for mistakes around position 21:
    [11:30:00] C1=CC=C(C=C1)OP(=O)((N%98-N=N-C(=C%98)C%9
    [11:30:00] ~~~~~~~~~~~~~~~~~~~~^
    [11:30:00] SMILES Parse Error: Failed parsing SMILES 'C1=CC=C(C=C1)OP(=O)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))OC2=CC=CC=C2' for input: 'C1=CC=C(C=C1)OP(=O)((N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))OC2=CC=CC=C2'
:::

::: {.output .display_data}
``` json
{"summary":"{\n  \"name\": \"total_df\",\n  \"rows\": 103,\n  \"fields\": [\n    {\n      \"column\": \"source\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 103,\n        \"samples\": [\n          \"C1C2CC3CC1CC(C2)(C3)N=[N+]=[N-]\",\n          \"COC1=CC=C(C=C1)COC(=O)N=[N+]=[N-]\",\n          \"C(CN=[N+]=[N-])C(=O)O\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"result\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 103,\n        \"samples\": [\n          \"C1C2CC3CC1CC(C2)(C3)(N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O)\",\n          \"COC1=CC=C(C=C1)COC(=O)(N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O)\",\n          \"C(C(N%98-N=N-C(=C%98)C%99=CC=C(C=C%99)C(C)C(=O)O))C(=O)O\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"total_df"}
```
:::
:::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="B494RsEqfAyk" outputId="ddc06404-52e4-4564-f150-d107b16ea6cf"}
``` python
from rdkit.Chem.Draw import MolsToGridImage
mols = [Chem.MolFromSmiles(smi) for smi in total_df["result"][:24]]
MolsToGridImage(mols)
```

::: {.output .execute_result execution_count="42"}
![](c99ea8a4eb058f57a73cbf3f7e9cd89e6d9dd10e.png)
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="QcLuMKfAgqHh" outputId="17e3194e-8e9d-4e9f-dbf0-c3364f3d04c1"}
``` python
from rdkit.Chem import AllChem
m3d=Chem.AddHs(mols[3])
Chem.AllChem.EmbedMolecule(m3d)
AllChem.MMFFOptimizeMolecule(m3d,maxIters=500,nonBondedThresh=200 )
```

::: {.output .execute_result execution_count="56"}
    0
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="R9nHKEtIgsyN" outputId="6f7a5406-db23-4c77-d866-134e494558b0"}
``` python
!pip install py3Dmol
```

::: {.output .stream .stdout}
    Requirement already satisfied: py3Dmol in /usr/local/lib/python3.12/dist-packages (2.5.3)
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":717}" id="QmUQSqIeblvi" outputId="8b6d4ddc-7ea5-475d-ff7f-7d71666231bd"}
``` python
import py3Dmol

mblock = Chem.MolToMolBlock(m3d)

view = py3Dmol.view(width=700, height=700)
view.addModel(mblock, 'mol')
view.setStyle({"stick":{}})
view.zoomTo()
view.show()
```

::: {.output .display_data}
<div id="3dmolviewer_17604420343409083"  style="position: relative; width: 700px; height: 700px;">
        <p id="3dmolwarning_17604420343409083" style="background-color:#ffcccc;color:black">3Dmol.js failed to load for some reason.  Please check your browser console for error messages.<br></p>
        </div>
<script>

var loadScriptAsync = function(uri){
  return new Promise((resolve, reject) => {
    //this is to ignore the existence of requirejs amd
    var savedexports, savedmodule;
    if (typeof exports !== 'undefined') savedexports = exports;
    else exports = {}
    if (typeof module !== 'undefined') savedmodule = module;
    else module = {}

    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {
        exports = savedexports;
        module = savedmodule;
        resolve();
    };
  var firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
});
};

if(typeof $3Dmolpromise === 'undefined') {
$3Dmolpromise = null;
  $3Dmolpromise = loadScriptAsync('https://cdn.jsdelivr.net/npm/3dmol@2.5.3/build/3Dmol-min.js');
}

var viewer_17604420343409083 = null;
var warn = document.getElementById("3dmolwarning_17604420343409083");
if(warn) {
    warn.parentNode.removeChild(warn);
}
$3Dmolpromise.then(function() {
viewer_17604420343409083 = $3Dmol.createViewer(document.getElementById("3dmolviewer_17604420343409083"),{backgroundColor:"white"});
viewer_17604420343409083.zoomTo();
	viewer_17604420343409083.addModel("\n     RDKit          3D\n\n 40 42  0  0  0  0  0  0  0  0999 V2000\n   -5.7626   -1.0299   -2.1565 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -5.5254   -1.4393   -0.8461 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.9108   -0.5711    0.0570 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.5230    0.7169   -0.3441 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.7699    1.1185   -1.6653 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -5.3860    0.2477   -2.5659 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -3.8592    1.6558    0.6326 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.5529    1.1746    1.0440 N   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.3288    0.8216    2.3220 N   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.0566    0.4585    2.4137 N   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.4738    0.5972    1.1767 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.4183    1.0686    0.2914 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.9197    0.2898    0.9019 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.4656    0.3930   -0.3838 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.8129    0.0913   -0.6171 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.6492   -0.3186    0.4306 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.1016   -0.4305    1.7151 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.7551   -0.1284    1.9462 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.1112   -0.6553    0.1969 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.8940    0.5132   -0.4080 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.2725   -1.9340   -0.6051 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.3298   -3.0641   -0.1439 O   0  0  0  0  0  0  0  0  0  0  0  0\n    5.3054   -1.7557   -1.9413 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -6.2413   -1.7077   -2.8587 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -5.8176   -2.4365   -0.5262 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.7301   -0.9092    1.0768 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.4861    2.1119   -2.0064 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -5.5737    0.5655   -3.5885 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -3.7068    2.6499    0.1968 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.4864    1.7777    1.5234 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.3979    1.3170   -0.7583 H   0  0  0  0  0  0  0  0  0  0  0  0\n    0.8563    0.7035   -1.2280 H   0  0  0  0  0  0  0  0  0  0  0  0\n    3.1988    0.1753   -1.6315 H   0  0  0  0  0  0  0  0  0  0  0  0\n    3.7142   -0.7573    2.5532 H   0  0  0  0  0  0  0  0  0  0  0  0\n    1.3625   -0.2261    2.9573 H   0  0  0  0  0  0  0  0  0  0  0  0\n    5.5878   -0.8608    1.1659 H   0  0  0  0  0  0  0  0  0  0  0  0\n    5.8402    1.3946    0.2406 H   0  0  0  0  0  0  0  0  0  0  0  0\n    5.5164    0.8048   -1.3940 H   0  0  0  0  0  0  0  0  0  0  0  0\n    6.9497    0.2464   -0.5306 H   0  0  0  0  0  0  0  0  0  0  0  0\n    5.3645   -2.6692   -2.2928 H   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  2  0\n  2  3  1  0\n  3  4  2  0\n  4  5  1  0\n  5  6  2  0\n  4  7  1  0\n  7  8  1  0\n  8  9  1  0\n  9 10  2  0\n 10 11  1  0\n 11 12  2  0\n 11 13  1  0\n 13 14  2  0\n 14 15  1  0\n 15 16  2  0\n 16 17  1  0\n 17 18  2  0\n 16 19  1  0\n 19 20  1  0\n 19 21  1  0\n 21 22  2  0\n 21 23  1  0\n  6  1  1  0\n 12  8  1  0\n 18 13  1  0\n  1 24  1  0\n  2 25  1  0\n  3 26  1  0\n  5 27  1  0\n  6 28  1  0\n  7 29  1  0\n  7 30  1  0\n 12 31  1  0\n 14 32  1  0\n 15 33  1  0\n 17 34  1  0\n 18 35  1  0\n 19 36  1  0\n 20 37  1  0\n 20 38  1  0\n 20 39  1  0\n 23 40  1  0\nM  END\n","mol");
	viewer_17604420343409083.setStyle({"stick": {}});
	viewer_17604420343409083.zoomTo();
viewer_17604420343409083.render();
});
</script>
:::
::::
