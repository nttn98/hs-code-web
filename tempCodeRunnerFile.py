import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

from hs_loader import load_hs_excels
from hs_search import find_candidate_rows
from model_client import ask_model_for_hs