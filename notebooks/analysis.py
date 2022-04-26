from progress.bar import Bar
from typing       import ( Dict, List, Optional as Maybe, Set, Tuple )
from scipy.stats  import entropy
# ------------------------------------------------------------------------------
import networkx as nx
import numpy    as np
# ------------------------------------------------------------------------------

first  = lambda pair: pair[0]
second = lambda pair: pair[1]

def second_moment( G : nx.Graph
                 ) -> float:
    """
    """
    return sum([d**2 for _,d in G.degree]) / G.number_of_nodes()


def imperfection( ranking_centrality : List[str]
                , ranking_sir        : List[str]
                , spreading_power    : Dict[str, float]
                , p
                ) -> float:
    """
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


def calculate_modularity_vitality_ranks_( G : nx.Graph
                                       , partition : List[Set[str]]
                                       , weight    : Maybe[str]     = None
                                       ) -> List[str]:
    modularity_base = nx.algorithms.community.quality.modularity( G
                                                                , communities = partition
                                                                , weight      = weight
                                                                )

    modularity_vitalities = dict()

    for node in Bar("Modularity Vitality", check_tty = False).iter(G.nodes):
        G_ = G.copy()
        G_.remove_node(node)
        partition_  = [ { n for n in module if n != node } for module in partition ]
        modularity_new = nx.algorithms.community.quality.modularity(G_, communities = partition_)
        modularity_vitalities[node] = abs(modularity_base - modularity_new)

    mod_vit_ranks = sorted( [ (k,v) for (k,v) in modularity_vitalities.items() ]
                      , key     = lambda p: p[1]
                      , reverse = True
                      )

    return [ node for node, _score in mod_vit_ranks ]


def calculate_modularity_vitality_ranks( G         : nx.Graph
                                       , partition : List[Set[str]]
                                       , weight    : Maybe[str]     = None
                                       ) -> List[str]:
    partition = partition.copy()
    G_ = G.copy()

    at_index = dict()
    for i,module in enumerate(partition):
        for node in module:
            at_index[node] = i

    modularity_base = nx.algorithms.community.quality.modularity( G_
                                                                , communities = partition
                                                                , weight      = None
                                                                )

    modularity_vitalities = dict()

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


def mkPartition(infomap):
    communities = dict()
    for node_ID, community in infomap.get_modules(depth_level = -1).items():
        if not community in communities:
            communities[community] = set()
        communities[community].add(infomap.get_name(node_ID))

    return communities


def mkCommunityAssignments(infomap):
    return { infomap.get_name(node_ID) : community for (node_ID, community) in infomap.get_modules(depth_level=-1).items() }


def showStats(G, community_assignments, communities):
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

    print(f"#nodes:       {G.number_of_nodes()}")
    print(f"#edges:       {G.number_of_edges()}")
    print(f"#communities: {len(communities)}")
    print(f"#eff. comm.:  {effective_number:.0f}")
    print(f"<k>:          {k1:.3f}")
    print(f"<k²>:         {k2:.3f}")
    print(f"transitivity: {nx.transitivity(G):.3f}")
    print(f"mixing:       {mixing:.3f}")
    print(f"λth:          {lam:.3f}")


def get_distribution_perplexity(communities, fraction, ranks):
    k : int = round(fraction * len(ranks))
    
    top_nodes = set(ranks[:k])
    h = entropy([ len(top_nodes.intersection(community)) / k for community in communities.values() ], base = 2)
    return 2**h


def toRanking(ranks, nodes):
    d = { u:rank for rank,u in enumerate(ranks) }
    return [ d[node] for node in nodes ]


def mkRanks(seq):
    return [str(u) for (u,_rank) in sorted(seq.items(), key = lambda p: p[1])]


def cascade(G, initial_active, threshold_gen):
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
