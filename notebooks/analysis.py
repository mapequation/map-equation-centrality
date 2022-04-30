from progress.bar import Bar
from typing       import ( Callable, Dict, Iterable, List, Optional as Maybe, Set, Tuple )
from scipy.stats  import entropy
# ------------------------------------------------------------------------------
import networkx as nx
import numpy    as np
# ------------------------------------------------------------------------------

# selectors for the first and second component of pairs
first  = lambda pair: pair[0]
second = lambda pair: pair[1]

def second_moment( G : nx.Graph
                 ) -> float:
    """
    Calculate the second moment of a graph's degree distribution.

    Parameters
    ----------
    G : nx.Graph
        The graph.

    Returns
    -------
    float
        The second moment of the graphs degree distribution.
    """
    return sum([d**2 for _,d in G.degree]) / G.number_of_nodes()


def imperfection( ranking_centrality : List[str]
                , ranking_sir        : List[str]
                , spreading_power    : Dict[str, float]
                , p                  : float
                ) -> float:
    """

    Parameters
    ----------
    ranking_centrality : List[str]
        A list of node labels that describes the node ranking under some centrality score.
    
    ranking_sir : List[str]
        A list of node labels that describes the node ranking as calculated with
        an SIR disease spreding simulation.

    spreading_power : Dict[str, float]
        A dictionary with node labels as keys and their spreading power as values.

    p : float
        Fraction of top-ranking nodes to select.
    
    Returns
    -------
    float
        The imperfection for the p-fraction of selected top-ranking nodes.
    """
    k : int = round(p * len(spreading_power))

    return 1 - ( np.average([ spreading_power[spreader] for spreader in ranking_centrality[:k] ])
               / np.average([ spreading_power[spreader] for spreader in ranking_sir[:k]        ])
               )


def get_imperfections( ranking_centrality : List[str]
                     , ranking_sir        : List[str]
                     , spreading_power    : Dict[str, float]
                     , node_fractions     : List[float]      = [ 0.02 * p for p in range(1,11) ]
                     ) -> List[Tuple[float, float]]:
    """
    Calculate imperfection values.

    Parameters
    ----------
    ranking_centrality : List[str]
        A list of node labels that describes the node ranking under some centrality score.

    ranking_sir : List[str]
        A list of node labels that describes the node ranking as calculated with
        an SIR disease spreding simulation.
    
    spreading_power : Dict[str, float]
        A dictionary with node labels as keys and their spreading power as values.
    
    node_fractions : List[float] = [ 0.02 * p for p in range(1,11) ]
        List of top-ranking node fractions for which the imperfection values should be calculated.
    
    Returns
    -------
    List[Tuple[float, float]]
        List containing pairs where the first component is the selected fraction of
        top-ranked nodes, and the second component the corresponding imperfection.
    """
    return [ (p, imperfection( ranking_centrality
                             , ranking_sir
                             , spreading_power
                             , p = p
                             ))
                 for p in node_fractions
           ]


def calculate_intra_inter_neighbours( G                     : nx.Graph
                                    , community_assignments : Dict[str, int]
                                    ) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Calculate the number of intra- and inter-module neighbours for all
    nodes in given graph under the given community assignemnts.

    Parameters
    ----------
    G: the graph

    community_assignments: the nodes' community assignments

    Returns
    -------
    A pair with two dictionaries where the first dictionary contains
    node labels as keys and their number of intra-module neighbours
    as values, and the second dictionary contains node labels as key
    and their number of inter-module neighbours as values.
    """
    intra_neighbours = dict()
    inter_neighbours = dict()

    for u in G.nodes:
        intra_neighbours[u] = 0
        inter_neighbours[u] = 0

        for v in G.neighbors(u):
            if community_assignments[u] == community_assignments[v]:
                intra_neighbours[u] += 1
            else:
                inter_neighbours[u] += 1

    return (intra_neighbours, inter_neighbours)


def calculate_community_hub_bridge_ranks( G                     : nx.Graph
                                        , communities           : Dict[int, Set[str]]
                                        , community_assignments : Dict[str, int]
                                        ) -> List[str] :
    """
    Calculate the ranking of the nodes in the given graph according
    to community hub-bridge under the given community structure.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    
    communities : Dict[int, Set[str]]
        A dictionary with community IDs as keys, and the set of nodes
        in each community as values.
    
    community_assignments : Dict[str, int]
        A dictionary with node labels as keys and the ID of the community
        to which they are assigned as values.
    
    Returns
    -------
    List[str]
        A list of node labes corresponding to the nodes' ranking according
        to community hub-bridge under the given community assignments.
    """
    intra_neighbours, inter_neighbours = calculate_intra_inter_neighbours( G = G
                                                                         , community_assignments = community_assignments
                                                                         )

    community_hub_bridge = dict()

    for u in Bar("Community Hub Bridge", check_tty = False).iter(G.nodes):
        c_k = len(communities[community_assignments[u]])
        NNC = { community_assignments[v] for v in G.neighbors(u) if community_assignments[v] != community_assignments[u] }

        community_hub_bridge[u] = c_k * intra_neighbours[u] + len(NNC) * inter_neighbours[u]

    com_hb_ranks = sorted(community_hub_bridge.items(), key = lambda p: p[1], reverse = True)

    return [ node for node, _score in com_hb_ranks ]


def calculate_community_based_centrality_ranks( G                     : nx.Graph
                                              , communities           : Dict[int, Set[str]]
                                              , community_assignments : Dict[str, int]
                                              ) -> List[str]:
    """
    Calculate the ranking of the nodes in the given graph according
    to community-based centrality under the given community structure.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    
    communities : Dict[int, Set[str]]
        A dictionary with community IDs as keys, and the set of nodes
        in each community as values.
    
    community_assignments : Dict[str, int]
        A dictionary with node labels as keys and the ID of the community
        to which they are assigned as values.
    
    Returns
    -------
    List[str]
        A list of node labes corresponding to the nodes' ranking according
        to community-based centrality under the given community assignments.
    """
    community_based_centrality = dict()

    community_factors = { community : len(community_nodes) / G.number_of_nodes()
                              for (community, community_nodes) in communities.items()
                        }

    for u in Bar("Community-based Centrality", check_tty = False).iter(G.nodes):
        community_neighbours = { community : 0 for community in communities }
        for v in G.neighbors(u):
            community_neighbours[community_assignments[v]] += 1
        community_based_centrality[u] = sum([community_neighbours[community] * community_factors[community]
                                                 for community in communities
                                            ])

    community_based_centrality_ranks = sorted(community_based_centrality.items(), key = lambda p: p[1], reverse = True)

    return [ node for node, _score in community_based_centrality_ranks ]


def calculate_modularity_vitality_ranks( G         : nx.Graph
                                       , partition : List[Set[str]]
                                       , weight    : Maybe[str]     = None
                                       ) -> List[str]:
    """
    Calculate the ranking of the nodes in the given graph according
    to modularity vitality under the given community structure.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    
    partition : List[Set[str]]
        The partition of the graph, represented as a list of sets where
        each set contains the labels of those nodes that are assigned to
        the respective community.
    
    weight : Maybe[str] = None
        The property that represents the links' weights.
    
    Returns
    -------
    List[str]
        A list of node labes corresponding to the nodes' ranking according
        to modularity vitality under the given community assignments.
    """

    # we will be destructive for efficiency reasons, so make a copy of
    # the graph and work with the copy!
    partition = partition.copy()
    G_ = G.copy()

    at_index = dict()
    for i,module in enumerate(partition):
        for node in module:
            at_index[node] = i

    # we first calculate the modularity for the complete graph
    modularity_base = nx.algorithms.community.quality.modularity( G_
                                                                , communities = partition
                                                                , weight      = weight
                                                                )

    modularity_vitalities = dict()

    # we calculate the modularity vitalities for each node in a somewhat hacky way:
    # we remove each node and remember what edges we remove along with it.
    # after calculating the new modularity, we put the deleted node and edges
    # back into the graph.
    for node in Bar("Modularity Vitality", check_tty = False).iter(list(G_.nodes)):
        # save nodes
        if G_.is_directed():
            deleted_in_edges  = [ u for u,_ in G_.in_edges(node)  ]
            deleted_out_edges = [ v for _,v in G_.out_edges(node) ]
        else:
            deleted_edges = list(G_.neighbors(node))

        G_.remove_node(node)
        partition[at_index[node]].remove(node)

        modularity_new = nx.algorithms.community.modularity(G_, communities = partition)
        modularity_vitalities[node] = abs(modularity_base - modularity_new)

        G_.add_node(node)
        partition[at_index[node]].add(node)

        if G_.is_directed():
            for u in deleted_in_edges:
                G_.add_edge(u,node)
            for v in deleted_out_edges:
                G_.add_edge(node,v)
        else:
            for u in deleted_edges:
                G_.add_edge(node, u)

    mod_vit_ranks = sorted( [ (k,v) for (k,v) in modularity_vitalities.items() ]
                      , key     = lambda p: p[1]
                      , reverse = True
                      )

    return [ node for node, _score in mod_vit_ranks ]


def mkPartition( infomap
               ) -> Dict[int, Set[str]]:
    """
    Extracts a partition from an infomap instance.

    Parameters
    ----------
    infomap 
        An infomap instance. We assume that it has been run so that
        we can actually extract a partition from it.

    Returns
    -------
    Dict[int, Set[str]]
        The networks partition calculated by the given infomap instance
        with community IDs as keys and sets of node labels as values.
        (Assuming that nodes were added with strings as IDs to Infomap.)
    """
    communities = dict()
    for node_ID, community in infomap.get_modules(depth_level = -1).items():
        if not community in communities:
            communities[community] = set()
        communities[community].add(infomap.get_name(node_ID))

    return communities


def mkCommunityAssignments( infomap
                          ) -> Dict[str, int]:
    """
    Extracts the nodes' community assignments from the given infomap instance.

    Parameters
    ----------
    infomap
        The infomap instance.

    Returns
    -------
    Dict[str, int]
        A dictionary with node labels as keys and the IDs of the communities
        they are assigned to as values. (Assuming that nodes were added with
        strings as IDs to Infomap.)
    """
    return { infomap.get_name(node_ID) : community for (node_ID, community) in infomap.get_modules(depth_level=-1).items() }


def showStats( G                     : nx.Graph
             , community_assignments : Dict[str, int]
             , communities           : Dict[int, Set[str]]
             ) -> None:
    """
    Calculates and prints some statistics.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    
    community_assignments : Dict[str, int]
        A dictionary with node labels as keys and the IDs of the communities
        they are assigned to as values.
    
    communities : Dict[int, Set[str]]
        A dictionary with community IDs as keys and the set of labels of those
        nodes assigned to each community as values.
    """

    intra_links = 0
    inter_links = 0

    for (u,v) in G.edges:
        if community_assignments[u] == community_assignments[v]:
            intra_links += 1
        else:
            inter_links += 1

    mixing = inter_links / (inter_links + intra_links)

    k1  = np.mean([d for _,d in G.degree])
    k2  = second_moment(G)
    lam = k1 / (k2-k1)

    effective_number = 2**entropy([len(m) for m in communities.values()], base = 2)

    print(f"#nodes:       {G.number_of_nodes()}")    # the number of nodes in the graph
    print(f"#edges:       {G.number_of_edges()}")    # the number of edges in the graph
    print(f"#communities: {len(communities)}")       # the number of communities
    print(f"#eff. comm.:  {effective_number:.0f}")   # the effective number of communities
    print(f"<k>:          {k1:.3f}")                 # the first moment of the degree distribution
    print(f"<k²>:         {k2:.3f}")                 # the second moment of the degree distribution
    print(f"transitivity: {nx.transitivity(G):.3f}") # the transitivity
    print(f"mixing:       {mixing:.3f}")             # the mixing
    print(f"λth:          {lam:.3f}")                # the epidemic threshold


def get_distribution_perplexity( communities : Dict[int, Set[str]]
                               , fraction    : float
                               , ranks       : List[str]
                               ) -> float:
    """
    Calculates the perplexity for a given fraction of top-ranked nodes
    under the given community structure.

    Parameters
    ----------
    communities : Dict[int, Set[str]]
        A dictionary with community IDs as keys and the set of labels of those
        nodes assigned to each community as values.
    
    fraction : float
        The fraction of top-ranked nodes to be selected.
    
    ranks : List[str]
        A list of node labels, describing the nodes' ranking.
    
    Returns
    -------
    float
        The perplexity of the given fraction of top-ranked nodes across the
        given communities.
    """
    k : int = round(fraction * len(ranks))
    
    top_nodes = set(ranks[:k])
    h = entropy([ len(top_nodes.intersection(community)) / k for community in communities.values() ], base = 2)
    return 2**h


def toRanking( ranks : List[str]
             , nodes : Iterable[str]
             ) -> List[int]:
    """
    Takes a list of node labels and converts it to a list of ranks in the same
    order as the nodes' labels are listed in `nodes`.

    Parameters
    ----------
    ranks : List[str]
        A list of node labels describing the nodes' ranking.

    nodes : Iterable[str]
        The sequence in which we wish to obtain the nodes' ranks.
    
    Returns
    -------
    List[int]
        List of nodes' ranks in the order as defined by `nodes`.
    """
    d = { u:rank for rank,u in enumerate(ranks) }
    return [ d[node] for node in nodes ]


def mkRanks( seq : Dict[str, int]
           ) -> List[str]:
    """
    Takes a dictionary with node labels as keys and their rank as values, and
    converts the ranking to a ranked list of noded labels.

    Parameters
    ----------
    seq : Dict[str, int]
        Dictionary with node labels as keys and their ranks as values.
    
    Returns
    -------
    List[str]
        The nodes' ranking.
    """
    return [str(u) for (u,_rank) in sorted(seq.items(), key = lambda p: p[1])]


def cascade( G              : nx.Graph
           , initial_active : List[str]
           , threshold_gen  : Callable
           ) -> float:
    """
    Simulates a cascade following the linear threshold model. Initially, a set
    of nodes is activated. In each step, we then activate those nodes whose
    fraction of active neighbouts is at least as high as their threshold.
    The per-node thresholds are given by the threshold generator. We stop when
    no more nodes get activated.

    G : nx.Graph
        The graph.
    
    initial_active : List[str]
        The list of initially active nodes.
    
    threshold_gen : Callable
        A function that returns node activation thresholds.
        (So far, its a constant.)

    Returns
    -------
    float
        The fraction of active nodes when the cascade ends.
    """
    # calculate thresholds
    thresholds = { u:threshold_gen() for u in G.nodes }
    
    # initially active and inactive nodes
    active     = set(initial_active)
    inactive   = set(G.nodes).difference(active)
    
    activating = active
    while len(activating) > 0:
        activating = set()
        for u in inactive:
            if G.is_directed():
                in_neighbours = set({ v for (v,u) in G.in_edges(u) })
                if len(in_neighbours) > 0 and len(active.intersection(in_neighbours)) / len(in_neighbours) >= thresholds[u]:
                    activating.add(u)

            else:
                if len(active.intersection(set(G.neighbors(u)))) / len(list(G.neighbors(u))) >= thresholds[u]:
                    activating.add(u)
        
        active   = active.union(activating)
        inactive = inactive.difference(activating)
    
    return len(active) / G.number_of_nodes()
