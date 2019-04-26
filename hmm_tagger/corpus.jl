module corpus
using Random
export load_tags, load_corpus, tag_index, unique_words, train_test_split, make_tag_frequencies, find_most_common_tag, evaluate_sentence, unigram_counts, bigram_counts, transition_prob, ngram_counts, viterbi

function load_tags(fname :: String)
    println("Loading tags from ", fname)
    tags = String[]
    open(fname) do f
        for ln in eachline(f)
            push!(tags, ln)
        end
    end
    push!(tags, "<s>")
    push!(tags, "</s>")
    tags
end

function tag_index(tag_list :: Array{String,1}, tag)
    findfirst(x->x==tag,tag_list)
end

function load_corpus(fname :: String, tags :: Array{String, 1})
    println("Loading corpus from ", fname)
    corpus = Array{Tuple{String, Int},1}[]
    open(fname) do f
        cur_sentence = Tuple{String, Int}[]
        new_sentence = true
        cur_id = "";
        for ln in eachline(f)
            if ln == ""
                push!(corpus,copy(cur_sentence))
                new_sentence = true
                empty!(cur_sentence)
            elseif new_sentence
                cur_id = ln
                new_sentence = false
            else
                tokens = split(ln)
                push!(cur_sentence, (tokens[1], tag_index(tags, tokens[2])))
            end
        end
    end
    corpus
end

function unique_words(corpus::Array{Array{Tuple{String, Int}, 1},1})
    words = Set{String}()
    for sen in corpus
        for (w,t) in sen
            push!(words, w)
        end
    end
    collect(words)
end

function train_test_split(array :: AbstractArray, at=0.7)
    idx = Random.randperm(length(array))
    last = Int(floor(length(array)*at))
    array[idx[1:last]], array[idx[last+1:end]]
end

function make_tag_frequencies(corpus, tags)
    dict = Dict{String, Array{Int, 1}}()
    for sen in corpus
        for (w, tag) in sen
            en = get(dict, w, zeros(Int, length(tags)))
            en[tag]+=1
            dict[w] = en
        end
    end
    ndict = Dict{String, Array{Float64, 1}}()
    for (k,v) in dict
        ndict[k] = v/sum(v)
    end
    ndict
end

function find_most_common_tag(frequencies)
    most_common_tag = Dict{String, Int}()
    for (k, v) in frequencies
        most_common_tag[k] = findmax(v)[2]
    end
    most_common_tag
end


function evaluate_sentence(sentence, tagger)
    correct = 0
    for (k, v) in sentence
        p = try tagger(k)
            catch e
            continue 
            #return (length(sentence), 0)
        end
        if v == p
            correct+=1
        end
    end
    (length(sentence), correct)
end

function unigram_counts(corpus, tags)
    counts = zeros(Int, length(tags))
    for sen in corpus
        for (w, t) in sen
            counts[t] += 1
        end
    end
    counts
end
function bigram_counts(corpus, tags)
    counts = Dict{Tuple{Int, Int}, Int}()
    for sen in corpus
        for i in 1:(length(sen)-1)
            bigram = (sen[i][2], sen[i+1][2])
            counts[bigram] = get(counts, bigram, 0) + 1 
        end
    end
    counts
end

function hmm_parameters(corpus, tags)
    ugc = zeros(Int, length(tags)) # unigram counts
    bgc = Dict{Tuple{Int, Int}, Int}() # bigram counts
    ecs = Dict{Tuple{String, Int}, Int}() # word-tag counts
    for sen in corpus
        previous = tag_index(tags, "<s>")
        ugc[previous]+=1
        for (w, t) in sen
            bgc[(previous, t)] = get(bgc, (previous, t), 0) + 1
            ugc[t] += 1
            previous = t
            ecs[(w,t)] = get(ecs, (w,t), 0) + 1
        end
        t = tag_index(tags, "</s>")
        bgc[(previous, t)] = get(bgc, (previous, t), 0) + 1
        ugc[t] += 1
    end
    t_prob = Dict{Tuple{Int, Int}, Float64}()
    for (k,v) in bgc
        t_prob[k] = v/ugc[k[1]]
    end
    e_prob = Dict{Tuple{String, Int}, Float64}()
    for (k,v) in ecs
        e_prob[k] = v/ugc[k[2]]
    end
    t_prob, e_prob
end




function transition_prob(unigram_ct, bigram_ct, new_state, old_state)
    get(bigram_ct, (old_state, new_state), 0)/unigram_ct[old_state]
end

function viterbi(t_probs, e_probs, states, initial_dist, observations,extra=false)
    vmat = zeros(length(states), length(observations))
    bmat = zeros(Int, length(states), length(observations))
    for s in 1:length(states)
        vmat[s,1] = initial_dist[s]*e_probs(observations[1],s)#get(e_probs, (observations[1], s), 0)
    end
    vmat[:,1] = vmat[:,1] / maximum(vmat[:,1]) # normalize
    for t in 2:length(observations)
        for s in 1:length(states)
            tmp = [vmat[sp, t-1]*get(t_probs, (sp,s),0) for sp in 1:length(states)]
            mi = argmax(tmp)
            mv = tmp[mi]
            vmat[s,t] = mv*e_probs(observations[t], s)#get(e_probs, (observations[t], s), 0)
            bmat[s,t] = mi
        end
        vmat[:,t] = vmat[:,t] / maximum(vmat[:,t]) # normalize
    end
    pathprob = maximum(vmat[:,end])
    path = [argmax(vmat[:,end])]
    for t in (length(observations)):-1:2
        push!(path, bmat[path[end], t])
    end
    if extra
        reverse(path), pathprob, vmat, bmat
    else
        reverse(path)
    end
end

end
