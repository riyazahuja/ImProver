/-- In a separated space, a complete set is closed. -/
theorem IsComplete.isClosed [UniformSpace α] [T0Space α] {s : Set α} (h : IsComplete s) :
    IsClosed s :=
  isClosed_iff_clusterPt.2 fun a ha => by
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      inst✝ : T0Space α
      s : Set α
      h : IsComplete s
      a : α
      ha : ClusterPt a (Filter.principal s)
      ⊢ Membership.mem s a
    -/
    let f := 𝓝[s] a
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      inst✝ : T0Space α
      s : Set α
      h : IsComplete s
      a : α
      ha : ClusterPt a (Filter.principal s)
      f : Filter α := nhdsWithin a s
      ⊢ Membership.mem s a
    -/
    have : Cauchy f := cauchy_nhds.mono' ha inf_le_left
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      inst✝ : T0Space α
      s : Set α
      h : IsComplete s
      a : α
      ha : ClusterPt a (Filter.principal s)
      f : Filter α := nhdsWithin a s
      this : Cauchy f
      ⊢ Membership.mem s a
    -/
    rcases h f this inf_le_right with ⟨y, ys, fy⟩
    /-
      case intro.intro
      α : Type u_1
      inst✝¹ : UniformSpace α
      inst✝ : T0Space α
      s : Set α
      h : IsComplete s
      a : α
      ha : ClusterPt a (Filter.principal s)
      f : Filter α := nhdsWithin a s
      this : Cauchy f
      y : α
      ys : Membership.mem s y
      fy : LE.le f (nhds y)
      ⊢ Membership.mem s a
    -/
    rwa [(tendsto_nhds_unique' ha inf_le_left fy : a = y)]
    /-
      🎉 no goals
    -/


theorem IsUniformEmbedding.isClosedEmbedding [UniformSpace α] [UniformSpace β] [CompleteSpace α]
    [T0Space β] {f : α → β} (hf : IsUniformEmbedding f) :
    IsClosedEmbedding f :=
  ⟨hf.isEmbedding, hf.isUniformInducing.isComplete_range.isClosed⟩


@[deprecated (since := "2024-10-30")]
alias IsUniformEmbedding.toIsClosedEmbedding := IsUniformEmbedding.isClosedEmbedding


@[deprecated (since := "2024-10-20")]
alias IsUniformEmbedding.toClosedEmbedding := IsUniformEmbedding.isClosedEmbedding


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.toIsClosedEmbedding := IsUniformEmbedding.isClosedEmbedding


@[deprecated (since := "2024-10-20")]
alias UniformEmbedding.toClosedEmbedding := IsUniformEmbedding.isClosedEmbedding


theorem continuous_extend_of_cauchy {e : α → β} {f : α → γ} (de : IsDenseInducing e)
    (h : ∀ b : β, Cauchy (map f (comap e <| 𝓝 b))) : Continuous (de.extend f) :=
  de.continuous_extend fun b => CompleteSpace.complete (h b)


