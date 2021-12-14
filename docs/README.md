### Documentation

To build the documentation, run the following commands from inside `docs/`.
First, generate the auto-documentation files 
```bash
sphinx-apidoc -f  -o "source/" "../dibs/" "../dibs/inference/dibs.py" "../dibs/utils"
```
This excludes all files after `"../dibs/"` from the documentation

After that, build the HTML 
```bash
make clean && make html
```
The documentation will be at `docs/build/html/index.html`.
