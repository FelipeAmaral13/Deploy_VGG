# Deploy_VGG

O objetivo desse projeto é fazer um deploy no streamlit usando um modelo já treinado de transfering learning em uma rede VGG19 para multi-classificação de folhas de uvas.

## Layout do deploy:
![Capturar](https://user-images.githubusercontent.com/5797933/174449894-a585e065-7e0e-4aeb-be41-8de504c9c7eb.PNG)


## Commandos uteis:

### Upar arquivos grandes no github:

1. Instalar o GitLFS ![link](https://git-lfs.github.com/)
2. Clonar o repositorio desejado
3. Add a extensão do arquivo considerado grande - `git lfs track “.fileextension”`
4. Copiar o arquivo para a pasta clonada
5. Add o arquivo no git - `git add filename.fileextension`
6. Se o arquivo for maior que 100Mb - `git lfs migrate import --include="*.fileextension"
