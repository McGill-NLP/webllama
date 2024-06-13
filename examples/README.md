# Examples

### Web API and client

You can find examples of how to use the server directly with `http.client.HTTPConnection` and through our client in [`examples/web_api/`](/examples/web_api/), respectively with `run_http.py` and `run_client.py`. You should let the server stay up for both examples. For more information, please read the section above about the Web API.

### End-to-end

You can find an end-to-end example of using `webllama.experimental` in [`examples/complete/run_all.py`](/examples/complete):

```bash
python examples/complete/run_all.py
```


### BrowserGym integration

We provide directly integration to BrowserGym and examples to use it. You can find an example at [`examples/browsergym/run_bg.py`](/examples/browsergym).


On remote server (with GPU and hosting the webllama model), run:
```bash
# transformers, sentence-transformers, pytorch, etc.
pip install -e .[modeling]
```

First, remotely, run:

```bash
# change if needed:
export CUDA_VISIBLE_DEVICES=0

python -m webllama.experimental.web.server --save_logs
```

Then, connect to your remote server via SSH:

```bash
# 8450 is the default port for our server
ssh -N -L 8450:localhost:8450 "<user>@<remote>"
```

Now, on your local machine, run:

```bash
pip install -e .
# browsergym integration
pip install "browsergym==0.3.*"
# install playwright
playwright install
```

```bash
python examples/browsergym/run_bg.py
```
