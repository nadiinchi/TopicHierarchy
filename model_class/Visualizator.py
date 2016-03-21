import numpy as np

class Visualizator: 
    """
    Creates web-navigator on hierarchy
    
    H is an instance of HAPTM class (built model)
    titles_file is a path to file in specific format containing docs info
    links_file is a path to file in specific format containing links info
    """
    def __init__(self, H, titles_file, links_file):
        self.pages = {}
        self.dictionary = H.dictionary
        self.read_files(titles_file, links_file)
        
        
    def read_files(self, titles_file, links_file):
        """
        reads files with info
        """
        self.titles = {}
        self.authors = {}
        self.links = {}
        with open(titles_file, "r") as fin:
            while True:
                doc = fin.readline()
                if not len(doc):
                    break
                doc = int(doc[:-1])
                authors = fin.readline()[:-1]
                title = fin.readline()[:-1]
                self.titles[doc] = title
                self.authors[doc] = authors
        with open(links_file, "r") as fin:
            for line in fin:
                doc, link = line[:-1].split("|")
                self.links[int(doc)] = link
                
        
    def fill(self, top_words_cnt=10, top_docs_cnt=100, psi_threshold=0.5):
        """
        collects topics info and creates graph of topics
        """
        theta = H.graph.theta
        for l in range(len(H.graph.phis)-1, -1, -1):
            antitops_words = np.argsort(H.graph.phis[l], axis=0)
            antitops_docs = np.argsort(theta, axis=1)   # *H.graph.eta[l][np.newaxis, :]
            for t in range(H.graph.phis[l].shape[1]):
                self.pages[(l, t)] = TopicPage(l, "Level "+str(l)+" Topic "+str(t),\
                                               "l"+str(l)+"t"+str(t)+".html",\
                                              [self.dictionary[w] for w in antitops_words[-1:-top_words_cnt-1:-1, t]],\
                                              [(self.titles[d], self.authors[d], self.links[d]) for d in \
                                              antitops_docs[t, -1:-top_docs_cnt-1:-1]], [], [])
            if l > 0:
                theta = H.graph.psis[l-1].dot(theta)
        
        for l in range(len(H.graph.psis)-1, -1, -1):
            for t in range(H.graph.psis[l].shape[0]):
                for s in range(H.graph.psis[l].shape[1]):
                    if H.graph.psis[l][t, s] > psi_threshold:
                        self.pages[(l, t)].children.append(self.pages[(l+1, s)])
                        self.pages[(l+1, s)].parents.append(self.pages[(l, t)])
          
                
                
    def print_pages(self, path):
        """
        prints pages to path
        """
        for key in self.pages.keys():
            self.pages[key].print_page(path)
        
        
class TopicPage:
    def __init__(self, level=0, title="", link="", top_words=[], top_docs=[], children=[], parents=[]):
        """
        Class for storing Visualizator nodes
        Args:
        level is int
        title is a string
        link is a string
        top_words is a list of words
        top_docs is a list of tuples (doc_title, doc_author, doc_link)
        children is a list of TopicPage instances
        parents is a list of TopicPage instances
        """
        self.level = level
        self.title = title
        self.link = link
        self.top_words = top_words
        self.top_docs = top_docs
        self.children = children
        self.parents = parents
        
        
    def print_page(self, path):
        with open(path+self.link, "w") as fout:
            fout.write("<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>"\
                '<html lang="en">\n<head>\n<meta content="text/html; charset=utf-8" http-equiv="Content-Type" />\n'\
                '<link media="all" href="style.css" type="text/css" rel="stylesheet" />\n'\
                "<title>"+self.title+"</title>\n"'<script src="script.js" type="text/javascript"></script>\n'\
                '</head>\n<body>\n')
            if self.level > 0:
                fout.write('<div class="parents">\n<h3> Надтемы: </h3>\n<ul>\n')
                for parent in self.parents:
                    fout.write('<li><a href="'+str(parent.link)+'"> '+str(parent.title)+' </a></li>\n')
                fout.write("</ul>\n</div>\n")
            else:
                fout.write('<div class="parents"> <br> </div>\n')
            fout.write('<div class="main_topic">\n')
            fout.write("<h1>"+self.title+"</h1>\n")
            fout.write("<p>"+", ".join(self.top_words)+"</p>\n")
            fout.write("<ul>")
            for title, author, link in self.top_docs:
                fout.write('<li><a href="'+link+'">'+title+" ("+author+") </a></li>\n")
            fout.write("</ul>\n</div>\n<br>\n")
            if self.children:
                width = 30 / float(len(self.children))
            fout.write('<div class="inline">\n')
            for child in self.children:
                fout.write('<div class="child_topic" style="width:'+str(width)+'cm;">\n')
                fout.write('<h2><a href="'+child.link+'">'+child.title+'</a></h2>\n')
                fout.write("<p>"+", ".join(child.top_words)+"</p>\n")
                fout.write("<ul>")
                for title, author, link in child.top_docs:
                    fout.write('<li><a href="'+link+'">'+title+" ("+author+") </a></li>\n")
                fout.write("</ul>\n</div>\n")
            fout.write("</div>\n</body>\n</html>")