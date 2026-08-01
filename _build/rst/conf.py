
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

project = "Dive into Deep Learning Personalized"
copyright = "2023, All authors. Licensed under CC-BY-SA-4.0 and MIT-0."
author = "Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola"
release = "1.0.3"

extensions = ["recommonmark","sphinxcontrib.bibtex","sphinxcontrib.rsvgconverter","sphinx.ext.autodoc","sphinx.ext.viewcode"]

# Customize bibtext_reference_style: https://github.com/mcmtroffaes/sphinxcontrib-bibtex/blob/develop/doc/usage.rst
from dataclasses import dataclass, field
import sphinxcontrib.bibtex.plugin

from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year \
    import AuthorYearReferenceStyle
def bracket_style() -> BracketStyle:
    return BracketStyle(
        left='(',
        right=')',
    )
@dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
    bracket_textual: BracketStyle = field(default_factory=bracket_style)
    bracket_author: BracketStyle = field(default_factory=bracket_style)
    bracket_label: BracketStyle = field(default_factory=bracket_style)
    bracket_year: BracketStyle = field(default_factory=bracket_style)
sphinxcontrib.bibtex.plugin.register_plugin(
    'sphinxcontrib.bibtex.style.referencing',
    'author_year_round', MyReferenceStyle)

# For original square brackets, just set bibtex_reference_style = 'author_year'
# If unspecified, citet output will be "Zhang et al. [Zhang et al. 2022]" instead of "Zhang et al. (2022)"
bibtex_reference_style = 'author_year_round'





templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

suppress_warnings = ['misc.highlighting_failure']
linkcheck_ignore = [r'.*localhost.*']
linkcheck_timeout = 5
linkcheck_workers = 20

autodoc_default_options = {
    'undoc-members': True,
    'show-inheritance': True,
}


html_theme = 'mxtheme'
html_theme_options = {
    'primary_color': 'blue',
    'accent_color': 'deep_orange',
    'header_links': [
        
    ],
    'show_footer': False
}
html_static_path = ['_static']

html_favicon = '_static/favicon.png'

html_logo = '_static/logo-with-text.png'

latex_documents = [
    (master_doc, "d2l-en.tex", "Dive into Deep Learning Personalized",
     author, 'PT1'),
]



rsvg_converter_args = ['-z', '0.8']

bibtex_bibfiles = ["d2l.bib"]

latex_engine = 'xelatex' # for utf-8 supports
latex_show_pagerefs = True
latex_show_urls = 'footnote'

latex_logo = '_static/logo.png'

latex_elements = {
'papersize':'a4paper,prodtf,twoside',
'figure_align': 'htbp',
'pointsize': '11pt',
'fvset':r'''\fvset{fontsize=\small}''',
'preamble': r'''
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsthm}
\usepackage{color}
\usepackage[figuresright]{rotating}
\usepackage{floatpag}
\rotfloatpagestyle{empty}
\usepackage{makeidx}
\usepackage{natbib}
\usepackage[parfill]{parskip}
\usepackage{titlesec}
\usepackage{multicol}
\usepackage{fixmath}
\protected\def\sphinxcite{\citep}

% Add bib to TOC
\usepackage[nottoc]{tocbibind}

% QR code sidenotes for all footnotes
% Make sure to replace special charactors URL Encoding: https://www.urlencoder.io/learn/
\usepackage{sidenotes}
\usepackage{marginfix}
\setlength\marginparpush{20pt}
\usepackage{qrcode}
\newcommand{\qrsidenote}[1]{
\sidenote{
\qrcode[height=8mm]{#1}}
}
\newcommand{\relaxfootnote}[1][]{}

\makeatletter
\let\ps@normal\ps@headings
\let\sphinxthebibliography\thebibliography
\let\sphinxtheindex\theindex
\let\sphinxAtStartFootnote\!
\let\footnote\relaxfootnote
\let\sphinxnolinkurl\qrsidenote

\sphinxDeclareColorOption{TitleColor}{{rgb}{0,0,0}}
\sphinxDeclareColorOption{InnerLinkColor}{{rgb}{0,0,0}}
\sphinxDeclareColorOption{OuterLinkColor}{{rgb}{0,0,0}}

% So some large pictures won't get the full page
\renewcommand{\floatpagefraction}{.8}

% Set the page margin size
\geometry{left=1.9in, right=1.4in, includefoot, bottom=2.25in, asymmetric} %%DAT changed bottom from 0.5in, add asymmetric

% Section and subsection style
\titleformat{\section}{\LARGE\centering\bfseries}{\thesection}%
            {0.5em}{}[{\hspace{-1.65in}\raggedleft\includegraphics[width=35pc]{PT1secrule.pdf}}]
\titleformat{\subsection}{\Large\centering\bfseries}%
            {\thesubsection}{0.5em}{}[{\color{gray}\titlerule[0.8pt]}]
\titleformat{\subsubsection}{\large\centering\bfseries}%
            {\thesubsubsection}{0.5em}{}[{\color{gray}}]

% Code font style, for more font style, visit: https://tug.org/FontCatalogue/
\setmonofont{Inconsolata}
%\renewcommand\ttfamily{\sffamily}

% Rewrite mathbf
% There are four parameters here: the encoding (OT1, T1, U), 
% the font family (cmss, Computer Modern Sans Serif, cmr: Roman), 
% the shape (n, i, b for bold, x for extended, or sometimes something else like small caps or upright italic) and the series. 
\DeclareMathAlphabet      {\newmathbf}{OT1}{cmr}{bx}{n}
\renewcommand{\mathbf}[1]{\newmathbf{#1}}
% Rewrite mathsf
\DeclareMathAlphabet      {\newmathsf}{OT1}{cmss}{m}{n}
\renewcommand{\mathsf}[1]{\newmathsf{#1}}
% Rewrite mathit
\DeclareMathAlphabet      {\newmathit}{OT1}{cmr}{m}{it}
\renewcommand{\mathit}[1]{\newmathit{#1}}

% Resize all figures
\let\ORIincludegraphics\includegraphics
\renewcommand{\includegraphics}[2][]{\ORIincludegraphics[scale=0.75,#1]{#2}}
% main text font style
%\usepackage{times}  %%DAT comment out and replace with newtx 
\PassOptionsToPackage{full}{textcomp}  %%Fixed by Aston - fix textcomp/newtxtext option clash issue
\usepackage{newtxtext}
\usepackage{newtxmath}

% Rewrite table of contents
\renewcommand\tableofcontents{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 %\AJW@addtocfalse
 \chapter*{\contentsname}
 %\@starttoc{toc}
 %\AJW@addtoctrue
 \if@restonecol\twocolumn\fi
  \@starttoc{toc}
}

\newcommand\cambridge{PT1}
\theoremstyle{plain}% default
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem*{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{condition}[theorem]{Condition}
\newtheorem{example-norules}[theorem]{Example}
\theoremstyle{remark}
\newtheorem*{remark}{Remark}
\newtheorem*{case}{Case}


\hyphenation{line-break line-breaks docu-ment triangle cambridge
    amsthdoc cambridgemods baseline-skip author authors
    cambridgestyle en-vir-on-ment polar astron-omers solu-tion}

\setcounter{tocdepth}{1}

\hbadness=99999  % or any number >=10000
\vfuzz=30pt
\hfuzz=30pt

% Defines macros for code-blocks styling
\definecolor{d2lbookOutputCellBackgroundColor}{RGB}{255,255,255}
\definecolor{d2lbookOutputCellBorderColor}{rgb}{.85,.85,.85}
\def\diilbookstyleoutputcell
   {\sphinxcolorlet{VerbatimColor}{d2lbookOutputCellBackgroundColor}%
    \sphinxcolorlet{VerbatimBorderColor}{d2lbookOutputCellBorderColor}%
    \sphinxsetup{verbatimwithframe,verbatimborder=0.5pt}%
   }%
%
\definecolor{d2lbookInputCellBackgroundColor}{rgb}{.95,.95,.95}
\def\diilbookstyleinputcell
   {\sphinxcolorlet{VerbatimColor}{d2lbookInputCellBackgroundColor}%
    \sphinxsetup{verbatimwithframe=false,verbatimborder=0pt}%
   }%


%DAT --BEGIN---
% To add the section and chapter to the odd page running head, 
% and to increase the indent in the toc to allow the subsection numbers more space

% add \thesection\enskip to recto r/h
\makeatletter
\def\ps@headings{%
\let\@mkboth=\markboth
\def\@evenfoot{}%
\def\@oddfoot {}%
\def\@evenhead{\hspace*{-5pc}\hbox to 5pc{%
\hfil\headandfootsize\seriffont\thepage\hfil\hspace{0.5pc}}%
\hbox to \textwidth{%
\hfill{\normalsize\seriffont%
{\leftmark}}\hfill}%
\hspace*{-35pc}{\thin@rule}%
}%
%
\def\@oddhead{\hspace*{-5pc}\hbox to 5pc{%
\hfil\headandfootsize\seriffont\thepage\hfil\hspace{0.5pc}}%
\hbox to \textwidth{%
\hfill\normalsize\seriffont
{\thesection\enskip\rightmark}\hfill}%ajw added \thesection\enskip
\hspace*{-35pc}{\thin@rule}%
}%
%\def\chaptermark##1{\markboth{##1}{##1}}%
%\def\sectionmark##1{\markright{\ifnum \c@secnumdepth >\z@
% \thesection\enskip\fi ##1}}%
\ifAJW@multisty
\def\chaptermark##1{}%
\def\sectionmark##1{}%
\else
\def\chaptermark##1{\markboth{##1}{##1}}%
\def\sectionmark##1{\markright{\ifnum \c@secnumdepth >\z@
\fi ##1}}%
\fi
}
\ps@headings% you must include this command to get the original overwritten
 
% change subsection tocindent 
\renewcommand*\l@subsection {\@dottedtocline{2}{38pt}{51pt}} % changed the second number (it was 31pt)
 
\makeatother
%DAT --END---


''',
'sphinxsetup': '''verbatimsep=2mm,
                  VerbatimColor={rgb}{.95,.95,.95},
                  VerbatimBorderColor={rgb}{.95,.95,.95},
                  pre_border-radius=3pt,
               ''',
'maketitle':'\\maketitle',
'tableofcontents': '\\tableofcontents',
'fncychap':'',
'makeindex':'\\makeindex'
}


latex_style_loc = "static/latex_style/"
latex_fnames = ["PT1/PT1.cls", "PT1header.eps", "PT1secrule.pdf", "PT1/PT1box.eps", "PT1/PT1chrule.eps", "PT1/multind.sty",  "PT1/amsthm.sty", "PT1/floatpag.sty", "PT1/rotating.sty", "PT1/myriad-pt1.sty", "PT1/natbib.sty",  "sphinxlatexstyleheadings.sty", "sphinxlatexstylepage.sty", "sphinxlatexindbibtoc.sty", "sphinxmessages.sty", "sphinxlatexobjects.sty", "PT1/natbib.dtx", "sphinxpackagefootnote.sty", "sphinxlatexlists.sty"]
latex_additional_files = [latex_style_loc + fname for fname in latex_fnames]



def setup(app):
    # app.add_js_file('https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js')
    app.add_js_file('d2l.js')
    app.add_css_file('d2l.css')
    import mxtheme
    app.add_directive('card', mxtheme.CardDirective)
