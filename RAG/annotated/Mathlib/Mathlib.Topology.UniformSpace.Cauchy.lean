/-- A filter `f` is Cauchy if for every entourage `r`, there exists an
  `s ∈ f` such that `s × s ⊆ r`. This is a generalization of Cauchy
  sequences, because if `a : ℕ → α` then the filter of sets containing
  cofinitely many of the `a n` is Cauchy iff `a` is a Cauchy sequence. -/
def Cauchy (f : Filter α) :=
  NeBot f ∧ f ×ˢ f ≤ 𝓤 α


/-- A set `s` is called *complete*, if any Cauchy filter `f` such that `s ∈ f`
has a limit in `s` (formally, it satisfies `f ≤ 𝓝 x` for some `x ∈ s`). -/
def IsComplete (s : Set α) :=
  ∀ f, Cauchy f → f ≤ 𝓟 s → ∃ x ∈ s, f ≤ 𝓝 x


theorem Filter.HasBasis.cauchy_iff {ι} {p : ι → Prop} {s : ι → Set (α × α)} (h : (𝓤 α).HasBasis p s)
    {f : Filter α} :
    Cauchy f ↔ NeBot f ∧ ∀ i, p i → ∃ t ∈ f, ∀ x ∈ t, ∀ y ∈ t, (x, y) ∈ s i :=
  and_congr Iff.rfl <|
    (f.basis_sets.prod_self.le_basis_iff h).trans <| by
      /-
        α : Type u
        uniformSpace : UniformSpace α
        ι : Sort u_1
        p : ι → Prop
        s : ι → Set (Prod α α)
        h : (uniformity α).HasBasis p s
        f : Filter α
        ⊢ Iff (∀ (i' : ι), p i' → Exists fun i => And (Membership.mem f i) (HasSubset. …
      -/
      simp only [subset_def, Prod.forall, mem_prod_eq, and_imp, id, forall_mem_comm]
      /-
        🎉 no goals
      -/


theorem cauchy_iff' {f : Filter α} :
    Cauchy f ↔ NeBot f ∧ ∀ s ∈ 𝓤 α, ∃ t ∈ f, ∀ x ∈ t, ∀ y ∈ t, (x, y) ∈ s :=
  (𝓤 α).basis_sets.cauchy_iff


theorem cauchy_iff {f : Filter α} : Cauchy f ↔ NeBot f ∧ ∀ s ∈ 𝓤 α, ∃ t ∈ f, t ×ˢ t ⊆ s :=
  cauchy_iff'.trans <| by
    /-
      α : Type u
      uniformSpace : UniformSpace α
      f : Filter α
      ⊢ Iff (And f.NeBot (∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s →  …
    -/
    simp only [subset_def, Prod.forall, mem_prod_eq, and_imp, id, forall_mem_comm]
    /-
      🎉 no goals
    -/


lemma cauchy_iff_le {l : Filter α} [hl : l.NeBot] :
    Cauchy l ↔ l ×ˢ l ≤ 𝓤 α := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    l : Filter α
    hl : l.NeBot
    ⊢ Iff (Cauchy l) (LE.le (SProd.sprod l l) (uniformity α))
  -/
  simp only [Cauchy, hl, true_and]
  /-
    🎉 no goals
  -/


theorem Cauchy.ultrafilter_of {l : Filter α} (h : Cauchy l) :
    Cauchy (@Ultrafilter.of _ l h.1 : Filter α) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    l : Filter α
    h : Cauchy l
    ⊢ Cauchy ↑(Ultrafilter.of l)
  -/
  haveI := h.1
  /-
    α : Type u
    uniformSpace : UniformSpace α
    l : Filter α
    h : Cauchy l
    this : l.NeBot
    ⊢ Cauchy ↑(Ultrafilter.of l)
  -/
  have := Ultrafilter.of_le l
  /-
    α : Type u
    uniformSpace : UniformSpace α
    l : Filter α
    h : Cauchy l
    this✝ : l.NeBot
    this : LE.le (↑(Ultrafilter.of l)) l
    ⊢ Cauchy ↑(Ultrafilter.of l)
  -/
  exact ⟨Ultrafilter.neBot _, (Filter.prod_mono this this).trans h.2⟩
  /-
    🎉 no goals
  -/


theorem cauchy_map_iff {l : Filter β} {f : β → α} :
    Cauchy (l.map f) ↔ NeBot l ∧ Tendsto (fun p : β × β => (f p.1, f p.2)) (l ×ˢ l) (𝓤 α) := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    l : Filter β
    f : β → α
    ⊢ Iff (Cauchy (Filter.map f l)) (And l.NeBot (Filter.Tendsto (fun p => { fst : …
  -/
  rw [Cauchy, map_neBot_iff, prod_map_map_eq, Tendsto]
  /-
    🎉 no goals
  -/


theorem cauchy_map_iff' {l : Filter β} [hl : NeBot l] {f : β → α} :
    Cauchy (l.map f) ↔ Tendsto (fun p : β × β => (f p.1, f p.2)) (l ×ˢ l) (𝓤 α) :=
  cauchy_map_iff.trans <| and_iff_right hl


theorem Cauchy.mono {f g : Filter α} [hg : NeBot g] (h_c : Cauchy f) (h_le : g ≤ f) : Cauchy g :=
  ⟨hg, le_trans (Filter.prod_mono h_le h_le) h_c.right⟩


theorem Cauchy.mono' {f g : Filter α} (h_c : Cauchy f) (_ : NeBot g) (h_le : g ≤ f) : Cauchy g :=
  h_c.mono h_le


theorem cauchy_nhds {a : α} : Cauchy (𝓝 a) :=
  ⟨nhds_neBot, nhds_prod_eq.symm.trans_le (nhds_le_uniformity a)⟩


theorem cauchy_pure {a : α} : Cauchy (pure a) :=
  cauchy_nhds.mono (pure_le_nhds a)


theorem Filter.Tendsto.cauchy_map {l : Filter β} [NeBot l] {f : β → α} {a : α}
    (h : Tendsto f l (𝓝 a)) : Cauchy (map f l) :=
  cauchy_nhds.mono h


lemma Cauchy.mono_uniformSpace {u v : UniformSpace β} {F : Filter β} (huv : u ≤ v)
    (hF : Cauchy (uniformSpace := u) F) : Cauchy (uniformSpace := v) F :=
  ⟨hF.1, hF.2.trans huv⟩


lemma cauchy_inf_uniformSpace {u v : UniformSpace β} {F : Filter β} :
    Cauchy (uniformSpace := u ⊓ v) F ↔
    Cauchy (uniformSpace := u) F ∧ Cauchy (uniformSpace := v) F := by
  /-
    β : Type v
    u v : UniformSpace β
    F : Filter β
    ⊢ Iff (Cauchy F) (And (Cauchy F) (Cauchy F))
  -/
  unfold Cauchy
  /-
    β : Type v
    u v : UniformSpace β
    F : Filter β
    ⊢ Iff (And F.NeBot (LE.le (SProd.sprod F F) (uniformity β))) (And (And F.NeBot …
  -/
  rw [inf_uniformity (u := u), le_inf_iff, and_and_left]
  /-
    🎉 no goals
  -/


lemma cauchy_iInf_uniformSpace {ι : Sort*} [Nonempty ι] {u : ι → UniformSpace β}
    {l : Filter β} :
    Cauchy (uniformSpace := ⨅ i, u i) l ↔ ∀ i, Cauchy (uniformSpace := u i) l := by
  /-
    β : Type v
    ι : Sort u_1
    inst✝ : Nonempty ι
    u : ι → UniformSpace β
    l : Filter β
    ⊢ Iff (Cauchy l) (∀ (i : ι), Cauchy l)
  -/
  unfold Cauchy
  /-
    β : Type v
    ι : Sort u_1
    inst✝ : Nonempty ι
    u : ι → UniformSpace β
    l : Filter β
    ⊢ Iff (And l.NeBot (LE.le (SProd.sprod l l) (uniformity β))) (∀ (i : ι), And l …
  -/
  rw [iInf_uniformity, le_iInf_iff, forall_and, forall_const]
  /-
    🎉 no goals
  -/


lemma cauchy_iInf_uniformSpace' {ι : Sort*} {u : ι → UniformSpace β}
    {l : Filter β} [l.NeBot] :
    Cauchy (uniformSpace := ⨅ i, u i) l ↔ ∀ i, Cauchy (uniformSpace := u i) l := by
  /-
    β : Type v
    ι : Sort u_1
    u : ι → UniformSpace β
    l : Filter β
    inst✝ : l.NeBot
    ⊢ Iff (Cauchy l) (∀ (i : ι), Cauchy l)
  -/
  simp_rw [cauchy_iff_le (uniformSpace := _), iInf_uniformity, le_iInf_iff]
  /-
    🎉 no goals
  -/


lemma cauchy_comap_uniformSpace {u : UniformSpace β} {α} {f : α → β} {l : Filter α} :
    Cauchy (uniformSpace := comap f u) l ↔ Cauchy (map f l) := by
  /-
    β : Type v
    u : UniformSpace β
    α : Type u_1
    f : α → β
    l : Filter α
    ⊢ Iff (Cauchy l) (Cauchy (Filter.map f l))
  -/
  simp only [Cauchy, map_neBot_iff, prod_map_map_eq, map_le_iff_le_comap]
  /-
    β : Type v
    u : UniformSpace β
    α : Type u_1
    f : α → β
    l : Filter α
    ⊢ Iff (And l.NeBot (LE.le (SProd.sprod l l) (uniformity α))) (And l.NeBot (LE. …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma cauchy_prod_iff [UniformSpace β] {F : Filter (α × β)} :
    Cauchy F ↔ Cauchy (map Prod.fst F) ∧ Cauchy (map Prod.snd F) := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    inst✝ : UniformSpace β
    F : Filter (Prod α β)
    ⊢ Iff (Cauchy F) (And (Cauchy (Filter.map Prod.fst F)) (Cauchy (Filter.map Pro …
  -/
  simp_rw [instUniformSpaceProd, ← cauchy_comap_uniformSpace, ← cauchy_inf_uniformSpace]
  /-
    🎉 no goals
  -/


theorem Cauchy.prod [UniformSpace β] {f : Filter α} {g : Filter β} (hf : Cauchy f) (hg : Cauchy g) :
    Cauchy (f ×ˢ g) := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    inst✝ : UniformSpace β
    f : Filter α
    g : Filter β
    hf : Cauchy f
    hg : Cauchy g
    ⊢ Cauchy (SProd.sprod f g)
  -/
  have := hf.1; have := hg.1
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    inst✝ : UniformSpace β
    f : Filter α
    g : Filter β
    hf : Cauchy f
    hg : Cauchy g
    this✝ : f.NeBot
    this : g.NeBot
    ⊢ Cauchy (SProd.sprod f g)
  -/
  simpa [cauchy_prod_iff, hf.1] using ⟨hf, hg⟩
  /-
    🎉 no goals
  -/


/-- The common part of the proofs of `le_nhds_of_cauchy_adhp` and
`SequentiallyComplete.le_nhds_of_seq_tendsto_nhds`: if for any entourage `s`
one can choose a set `t ∈ f` of diameter `s` such that it contains a point `y`
with `(x, y) ∈ s`, then `f` converges to `x`. -/
theorem le_nhds_of_cauchy_adhp_aux {f : Filter α} {x : α}
    (adhs : ∀ s ∈ 𝓤 α, ∃ t ∈ f, t ×ˢ t ⊆ s ∧ ∃ y, (x, y) ∈ s ∧ y ∈ t) : f ≤ 𝓝 x := by
  -- Consider a neighborhood `s` of `x`
  /-
    α : Type u
    uniformSpace : UniformSpace α
    f : Filter α
    x : α
    adhs : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t  …
    ⊢ LE.le f (nhds x)
  -/
  intro s hs
  -- Take an entourage twice smaller than `s`
  /-
    α : Type u
    uniformSpace : UniformSpace α
    f : Filter α
    x : α
    adhs : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t  …
    s : Set α
    hs : Membership.mem (nhds x) s
    ⊢ Membership.mem f s
  -/
  rcases comp_mem_uniformity_sets (mem_nhds_uniformity_iff_right.1 hs) with ⟨U, U_mem, hU⟩
  -- Take a set `t ∈ f`, `t × t ⊆ U`, and a point `y ∈ t` such that `(x, y) ∈ U`
  /-
    case intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    f : Filter α
    x : α
    adhs : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t  …
    s : Set α
    hs : Membership.mem (nhds x) s
    U : Set (Prod α α)
    U_mem : Membership.mem (uniformity α) U
    hU : HasSubset.Subset (compRel U U) (setOf fun p => Eq p.1 x → Membership.mem  …
    ⊢ Membership.mem f s
  -/
  rcases adhs U U_mem with ⟨t, t_mem, ht, y, hxy, hy⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    f : Filter α
    x : α
    adhs : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t  …
    s : Set α
    hs : Membership.mem (nhds x) s
    U : Set (Prod α α)
    U_mem : Membership.mem (uniformity α) U
    hU : HasSubset.Subset (compRel U U) (setOf fun p => Eq p.1 x → Membership.mem  …
    t : Set α
    t_mem : Membership.mem f t
    ht : HasSubset.Subset (SProd.sprod t t) U
    y : α
    hxy : Membership.mem U { fst := x, snd := y }
    hy : Membership.mem t y
    ⊢ Membership.mem f s
  -/
  apply mem_of_superset t_mem
  -- Given a point `z ∈ t`, we have `(x, y) ∈ U` and `(y, z) ∈ t × t ⊆ U`, hence `z ∈ s`
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    f : Filter α
    x : α
    adhs : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t  …
    s : Set α
    hs : Membership.mem (nhds x) s
    U : Set (Prod α α)
    U_mem : Membership.mem (uniformity α) U
    hU : HasSubset.Subset (compRel U U) (setOf fun p => Eq p.1 x → Membership.mem  …
    t : Set α
    t_mem : Membership.mem f t
    ht : HasSubset.Subset (SProd.sprod t t) U
    y : α
    hxy : Membership.mem U { fst := x, snd := y }
    hy : Membership.mem t y
    ⊢ HasSubset.Subset t s
  -/
  exact fun z hz => hU (prod_mk_mem_compRel hxy (ht <| mk_mem_prod hy hz)) rfl
  /-
    🎉 no goals
  -/


/-- If `x` is an adherent (cluster) point for a Cauchy filter `f`, then it is a limit point
for `f`. -/
theorem le_nhds_of_cauchy_adhp {f : Filter α} {x : α} (hf : Cauchy f) (adhs : ClusterPt x f) :
    f ≤ 𝓝 x :=
  le_nhds_of_cauchy_adhp_aux
    (fun s hs => by
      /-
        α : Type u
        uniformSpace : UniformSpace α
        f : Filter α
        x : α
        hf : Cauchy f
        adhs : ClusterPt x f
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        ⊢ Exists fun t => And (Membership.mem f t) (And (HasSubset.Subset (SProd.sprod …
      -/
      obtain ⟨t, t_mem, ht⟩ : ∃ t ∈ f, t ×ˢ t ⊆ s := (cauchy_iff.1 hf).2 s hs
      /-
        case intro.intro
        α : Type u
        uniformSpace : UniformSpace α
        f : Filter α
        x : α
        hf : Cauchy f
        adhs : ClusterPt x f
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        t : Set α
        t_mem : Membership.mem f t
        ht : HasSubset.Subset (SProd.sprod t t) s
        ⊢ Exists fun t => And (Membership.mem f t) (And (HasSubset.Subset (SProd.sprod …
      -/
      use t, t_mem, ht
      /-
        case right
        α : Type u
        uniformSpace : UniformSpace α
        f : Filter α
        x : α
        hf : Cauchy f
        adhs : ClusterPt x f
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        t : Set α
        t_mem : Membership.mem f t
        ht : HasSubset.Subset (SProd.sprod t t) s
        ⊢ Exists fun y => And (Membership.mem s { fst := x, snd := y }) (Membership.me …
      -/
      exact forall_mem_nonempty_iff_neBot.2 adhs _ (inter_mem_inf (mem_nhds_left x hs) t_mem))
      /-
        🎉 no goals
      -/


theorem le_nhds_iff_adhp_of_cauchy {f : Filter α} {x : α} (hf : Cauchy f) :
    f ≤ 𝓝 x ↔ ClusterPt x f :=
  ⟨fun h => ClusterPt.of_le_nhds' h hf.1, le_nhds_of_cauchy_adhp hf⟩


nonrec theorem Cauchy.map [UniformSpace β] {f : Filter α} {m : α → β} (hf : Cauchy f)
    (hm : UniformContinuous m) : Cauchy (map m f) :=
  ⟨hf.1.map _,
    calc
      map m f ×ˢ map m f = map (Prod.map m m) (f ×ˢ f) := Filter.prod_map_map_eq
      _ ≤ Filter.map (Prod.map m m) (𝓤 α) := map_mono hf.right
      _ ≤ 𝓤 β := hm⟩


nonrec theorem Cauchy.comap [UniformSpace β] {f : Filter β} {m : α → β} (hf : Cauchy f)
    (hm : comap (fun p : α × α => (m p.1, m p.2)) (𝓤 β) ≤ 𝓤 α) [NeBot (comap m f)] :
    Cauchy (comap m f) :=
  ⟨‹_›,
    calc
      comap m f ×ˢ comap m f = comap (Prod.map m m) (f ×ˢ f) := prod_comap_comap_eq
      _ ≤ comap (Prod.map m m) (𝓤 β) := comap_mono hf.right
      _ ≤ 𝓤 α := hm⟩


theorem Cauchy.comap' [UniformSpace β] {f : Filter β} {m : α → β} (hf : Cauchy f)
    (hm : Filter.comap (fun p : α × α => (m p.1, m p.2)) (𝓤 β) ≤ 𝓤 α)
    (_ : NeBot (Filter.comap m f)) : Cauchy (Filter.comap m f) :=
  hf.comap hm


/-- Cauchy sequences. Usually defined on ℕ, but often it is also useful to say that a function
defined on ℝ is Cauchy at +∞ to deduce convergence. Therefore, we define it in a type class that
is general enough to cover both ℕ and ℝ, which are the main motivating examples. -/
def CauchySeq [Preorder β] (u : β → α) :=
  Cauchy (atTop.map u)


theorem CauchySeq.tendsto_uniformity [Preorder β] {u : β → α} (h : CauchySeq u) :
    Tendsto (Prod.map u u) atTop (𝓤 α) := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    inst✝ : Preorder β
    u : β → α
    h : CauchySeq u
    ⊢ Filter.Tendsto (Prod.map u u) Filter.atTop (uniformity α)
  -/
  simpa only [Tendsto, prod_map_map_eq', prod_atTop_atTop_eq] using h.right
  /-
    🎉 no goals
  -/


theorem CauchySeq.nonempty [Preorder β] {u : β → α} (hu : CauchySeq u) : Nonempty β :=
  @nonempty_of_neBot _ _ <| (map_neBot_iff _).1 hu.1


theorem CauchySeq.mem_entourage {β : Type*} [SemilatticeSup β] {u : β → α} (h : CauchySeq u)
    {V : Set (α × α)} (hV : V ∈ 𝓤 α) : ∃ k₀, ∀ i j, k₀ ≤ i → k₀ ≤ j → (u i, u j) ∈ V := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    β : Type u_1
    inst✝ : SemilatticeSup β
    u : β → α
    h : CauchySeq u
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    ⊢ Exists fun k₀ => ∀ (i j : β), LE.le k₀ i → LE.le k₀ j → Membership.mem V { f …
  -/
  haveI := h.nonempty
  /-
    α : Type u
    uniformSpace : UniformSpace α
    β : Type u_1
    inst✝ : SemilatticeSup β
    u : β → α
    h : CauchySeq u
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    this : Nonempty β
    ⊢ Exists fun k₀ => ∀ (i j : β), LE.le k₀ i → LE.le k₀ j → Membership.mem V { f …
  -/
  have := h.tendsto_uniformity; rw [← prod_atTop_atTop_eq] at this
  /-
    α : Type u
    uniformSpace : UniformSpace α
    β : Type u_1
    inst✝ : SemilatticeSup β
    u : β → α
    h : CauchySeq u
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    this✝ : Nonempty β
    this : Filter.Tendsto (Prod.map u u) (SProd.sprod Filter.atTop Filter.atTop) ( …
    ⊢ Exists fun k₀ => ∀ (i j : β), LE.le k₀ i → LE.le k₀ j → Membership.mem V { f …
  -/
  simpa [MapsTo] using atTop_basis.prod_self.tendsto_left_iff.1 this V hV
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.cauchySeq [SemilatticeSup β] [Nonempty β] {f : β → α} {x}
    (hx : Tendsto f atTop (𝓝 x)) : CauchySeq f :=
  hx.cauchy_map


theorem cauchySeq_const [SemilatticeSup β] [Nonempty β] (x : α) : CauchySeq fun _ : β => x :=
  tendsto_const_nhds.cauchySeq


theorem cauchySeq_iff_tendsto [Nonempty β] [SemilatticeSup β] {u : β → α} :
    CauchySeq u ↔ Tendsto (Prod.map u u) atTop (𝓤 α) :=
                              /-
                                α : Type u
                                β : Type v
                                uniformSpace : UniformSpace α
                                inst✝¹ : Nonempty β
                                inst✝ : SemilatticeSup β
                                u : β → α
                                ⊢ Iff (Filter.Tendsto (fun p => { fst := u p.1, snd := u p.2 }) (SProd.sprod F …
                              -/
  cauchy_map_iff'.trans <| by simp only [prod_atTop_atTop_eq, Prod.map_def]
                              /-
                                🎉 no goals
                              -/


theorem CauchySeq.comp_tendsto {γ} [Preorder β] [SemilatticeSup γ] [Nonempty γ] {f : β → α}
    (hf : CauchySeq f) {g : γ → β} (hg : Tendsto g atTop atTop) : CauchySeq (f ∘ g) :=
  ⟨inferInstance, le_trans (prod_le_prod.mpr ⟨Tendsto.comp le_rfl hg, Tendsto.comp le_rfl hg⟩) hf.2⟩


theorem CauchySeq.comp_injective [SemilatticeSup β] [NoMaxOrder β] [Nonempty β] {u : ℕ → α}
    (hu : CauchySeq u) {f : β → ℕ} (hf : Injective f) : CauchySeq (u ∘ f) :=
  hu.comp_tendsto <| Nat.cofinite_eq_atTop ▸ hf.tendsto_cofinite.mono_left atTop_le_cofinite


theorem Function.Bijective.cauchySeq_comp_iff {f : ℕ → ℕ} (hf : Bijective f) (u : ℕ → α) :
    CauchySeq (u ∘ f) ↔ CauchySeq u := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    f : Nat → Nat
    hf : Function.Bijective f
    u : Nat → α
    ⊢ Iff (CauchySeq (Function.comp u f)) (CauchySeq u)
  -/
  refine ⟨fun H => ?_, fun H => H.comp_injective hf.injective⟩
  /-
    α : Type u
    uniformSpace : UniformSpace α
    f : Nat → Nat
    hf : Function.Bijective f
    u : Nat → α
    H : CauchySeq (Function.comp u f)
    ⊢ CauchySeq u
  -/
  lift f to ℕ ≃ ℕ using hf
  /-
    case intro
    α : Type u
    uniformSpace : UniformSpace α
    u : Nat → α
    f : Equiv Nat Nat
    H : CauchySeq (Function.comp u ⇑f)
    ⊢ CauchySeq u
  -/
  simpa only [Function.comp_def, f.apply_symm_apply] using H.comp_injective f.symm.injective
  /-
    🎉 no goals
  -/


theorem CauchySeq.subseq_subseq_mem {V : ℕ → Set (α × α)} (hV : ∀ n, V n ∈ 𝓤 α) {u : ℕ → α}
    (hu : CauchySeq u) {f g : ℕ → ℕ} (hf : Tendsto f atTop atTop) (hg : Tendsto g atTop atTop) :
    ∃ φ : ℕ → ℕ, StrictMono φ ∧ ∀ n, ((u ∘ f ∘ φ) n, (u ∘ g ∘ φ) n) ∈ V n := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    V : Nat → Set (Prod α α)
    hV : ∀ (n : Nat), Membership.mem (uniformity α) (V n)
    u : Nat → α
    hu : CauchySeq u
    f g : Nat → Nat
    hf : Filter.Tendsto f Filter.atTop Filter.atTop
    hg : Filter.Tendsto g Filter.atTop Filter.atTop
    ⊢ Exists fun φ => And (StrictMono φ) (∀ (n : Nat), Membership.mem (V n) { fst  …
  -/
  rw [cauchySeq_iff_tendsto] at hu
  /-
    α : Type u
    uniformSpace : UniformSpace α
    V : Nat → Set (Prod α α)
    hV : ∀ (n : Nat), Membership.mem (uniformity α) (V n)
    u : Nat → α
    hu : Filter.Tendsto (Prod.map u u) Filter.atTop (uniformity α)
    f g : Nat → Nat
    hf : Filter.Tendsto f Filter.atTop Filter.atTop
    hg : Filter.Tendsto g Filter.atTop Filter.atTop
    ⊢ Exists fun φ => And (StrictMono φ) (∀ (n : Nat), Membership.mem (V n) { fst  …
  -/
  exact ((hu.comp <| hf.prod_atTop hg).comp tendsto_atTop_diagonal).subseq_mem hV
  /-
    🎉 no goals
  -/

-- todo: generalize this and other lemmas to a nonempty semilattice

theorem cauchySeq_iff' {u : ℕ → α} :
    CauchySeq u ↔ ∀ V ∈ 𝓤 α, ∀ᶠ k in atTop, k ∈ Prod.map u u ⁻¹' V :=
  cauchySeq_iff_tendsto


theorem cauchySeq_iff {u : ℕ → α} :
    CauchySeq u ↔ ∀ V ∈ 𝓤 α, ∃ N, ∀ k ≥ N, ∀ l ≥ N, (u k, u l) ∈ V := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    u : Nat → α
    ⊢ Iff (CauchySeq u) (∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → …
  -/
  simp only [cauchySeq_iff', Filter.eventually_atTop_prod_self', mem_preimage, Prod.map_apply]
  /-
    🎉 no goals
  -/


theorem CauchySeq.prod_map {γ δ} [UniformSpace β] [Preorder γ] [Preorder δ] {u : γ → α}
    {v : δ → β} (hu : CauchySeq u) (hv : CauchySeq v) : CauchySeq (Prod.map u v) := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    γ : Type u_1
    δ : Type u_2
    inst✝² : UniformSpace β
    inst✝¹ : Preorder γ
    inst✝ : Preorder δ
    u : γ → α
    v : δ → β
    hu : CauchySeq u
    hv : CauchySeq v
    ⊢ CauchySeq (Prod.map u v)
  -/
  simpa only [CauchySeq, prod_map_map_eq', prod_atTop_atTop_eq] using hu.prod hv
  /-
    🎉 no goals
  -/


theorem CauchySeq.prod {γ} [UniformSpace β] [Preorder γ] {u : γ → α} {v : γ → β}
    (hu : CauchySeq u) (hv : CauchySeq v) : CauchySeq fun x => (u x, v x) :=
  haveI := hu.1.of_map
  (Cauchy.prod hu hv).mono (Tendsto.prod_mk le_rfl le_rfl)


theorem CauchySeq.eventually_eventually [SemilatticeSup β] {u : β → α} (hu : CauchySeq u)
    {V : Set (α × α)} (hV : V ∈ 𝓤 α) : ∀ᶠ k in atTop, ∀ᶠ l in atTop, (u k, u l) ∈ V :=
  eventually_atTop_curry <| hu.tendsto_uniformity hV


theorem UniformContinuous.comp_cauchySeq {γ} [UniformSpace β] [Preorder γ] {f : α → β}
    (hf : UniformContinuous f) {u : γ → α} (hu : CauchySeq u) : CauchySeq (f ∘ u) :=
  hu.map hf


theorem CauchySeq.subseq_mem {V : ℕ → Set (α × α)} (hV : ∀ n, V n ∈ 𝓤 α) {u : ℕ → α}
    (hu : CauchySeq u) : ∃ φ : ℕ → ℕ, StrictMono φ ∧ ∀ n, (u <| φ (n + 1), u <| φ n) ∈ V n := by
  have : ∀ n, ∃ N, ∀ k ≥ N, ∀ l ≥ k, (u l, u k) ∈ V n := fun n => by
    rw [cauchySeq_iff] at hu
    rcases hu _ (hV n) with ⟨N, H⟩
    exact ⟨N, fun k hk l hl => H _ (le_trans hk hl) _ hk⟩
  obtain ⟨φ : ℕ → ℕ, φ_extr : StrictMono φ, hφ : ∀ n, ∀ l ≥ φ n, (u l, u <| φ n) ∈ V n⟩ :=
    extraction_forall_of_eventually' this
  /-
    case intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    V : Nat → Set (Prod α α)
    hV : ∀ (n : Nat), Membership.mem (uniformity α) (V n)
    u : Nat → α
    hu : CauchySeq u
    this : ∀ (n : Nat), Exists fun N => ∀ (k : Nat), GE.ge k N → ∀ (l : Nat), GE.g …
    φ : Nat → Nat
    φ_extr : StrictMono φ
    hφ : ∀ (n l : Nat), GE.ge l (φ n) → Membership.mem (V n) { fst := u l, snd :=  …
    ⊢ Exists fun φ => And (StrictMono φ) (∀ (n : Nat), Membership.mem (V n) { fst  …
  -/
  exact ⟨φ, φ_extr, fun n => hφ _ _ (φ_extr <| Nat.lt_add_one n).le⟩
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.subseq_mem_entourage {V : ℕ → Set (α × α)} (hV : ∀ n, V n ∈ 𝓤 α) {u : ℕ → α}
    {a : α} (hu : Tendsto u atTop (𝓝 a)) : ∃ φ : ℕ → ℕ, StrictMono φ ∧ (u (φ 0), a) ∈ V 0 ∧
      ∀ n, (u <| φ (n + 1), u <| φ n) ∈ V (n + 1) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    V : Nat → Set (Prod α α)
    hV : ∀ (n : Nat), Membership.mem (uniformity α) (V n)
    u : Nat → α
    a : α
    hu : Filter.Tendsto u Filter.atTop (nhds a)
    ⊢ Exists fun φ => And (StrictMono φ) (And (Membership.mem (V 0) { fst := u (φ  …
  -/
  rcases mem_atTop_sets.1 (hu (ball_mem_nhds a (symm_le_uniformity <| hV 0))) with ⟨n, hn⟩
  rcases (hu.comp (tendsto_add_atTop_nat n)).cauchySeq.subseq_mem fun n => hV (n + 1) with
    ⟨φ, φ_mono, hφV⟩
  /-
    case intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    V : Nat → Set (Prod α α)
    hV : ∀ (n : Nat), Membership.mem (uniformity α) (V n)
    u : Nat → α
    a : α
    hu : Filter.Tendsto u Filter.atTop (nhds a)
    n : Nat
    hn : ∀ (b : Nat), GE.ge b n → Membership.mem (Set.preimage u (UniformSpace.bal …
    φ : Nat → Nat
    φ_mono : StrictMono φ
    hφV : ∀ (n_1 : Nat), Membership.mem (V (HAdd.hAdd n_1 1)) { fst := Function.co …
    ⊢ Exists fun φ => And (StrictMono φ) (And (Membership.mem (V 0) { fst := u (φ  …
  -/
  exact ⟨fun k => φ k + n, φ_mono.add_const _, hn _ le_add_self, hφV⟩
  /-
    🎉 no goals
  -/


/-- If a Cauchy sequence has a convergent subsequence, then it converges. -/
theorem tendsto_nhds_of_cauchySeq_of_subseq [Preorder β] {u : β → α} (hu : CauchySeq u)
    {ι : Type*} {f : ι → β} {p : Filter ι} [NeBot p] (hf : Tendsto f p atTop) {a : α}
    (ha : Tendsto (u ∘ f) p (𝓝 a)) : Tendsto u atTop (𝓝 a) :=
  le_nhds_of_cauchy_adhp hu (ha.mapClusterPt.of_comp hf)


/-- Any shift of a Cauchy sequence is also a Cauchy sequence. -/
theorem cauchySeq_shift {u : ℕ → α} (k : ℕ) : CauchySeq (fun n ↦ u (n + k)) ↔ CauchySeq u := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    u : Nat → α
    k : Nat
    ⊢ Iff (CauchySeq fun n => u (HAdd.hAdd n k)) (CauchySeq u)
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h : CauchySeq fun n => u (HAdd.hAdd n k)
      ⊢ CauchySeq u
    -/
  · rw [cauchySeq_iff] at h ⊢
    /-
      case mp
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h : ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → Exists fun N =>  …
      ⊢ ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → Exists fun N => ∀  …
    -/
    intro V mV
    /-
      case mp
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h : ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → Exists fun N =>  …
      V : Set (Prod α α)
      mV : Membership.mem (uniformity α) V
      ⊢ Exists fun N => ∀ (k : Nat), GE.ge k N → ∀ (l : Nat), GE.ge l N → Membership …
    -/
    obtain ⟨N, h⟩ := h V mV
    /-
      case mp.intro
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h✝ : ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → Exists fun N => …
      V : Set (Prod α α)
      mV : Membership.mem (uniformity α) V
      N : Nat
      h : ∀ (k_1 : Nat), GE.ge k_1 N → ∀ (l : Nat), GE.ge l N → Membership.mem V { f …
      ⊢ Exists fun N => ∀ (k : Nat), GE.ge k N → ∀ (l : Nat), GE.ge l N → Membership …
    -/
    use N + k
    /-
      case h
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h✝ : ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → Exists fun N => …
      V : Set (Prod α α)
      mV : Membership.mem (uniformity α) V
      N : Nat
      h : ∀ (k_1 : Nat), GE.ge k_1 N → ∀ (l : Nat), GE.ge l N → Membership.mem V { f …
      ⊢ ∀ (k_1 : Nat), GE.ge k_1 (HAdd.hAdd N k) → ∀ (l : Nat), GE.ge l (HAdd.hAdd N …
    -/
    intro a ha b hb
    /-
      case h
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h✝ : ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → Exists fun N => …
      V : Set (Prod α α)
      mV : Membership.mem (uniformity α) V
      N : Nat
      h : ∀ (k_1 : Nat), GE.ge k_1 N → ∀ (l : Nat), GE.ge l N → Membership.mem V { f …
      a : Nat
      ha : GE.ge a (HAdd.hAdd N k)
      b : Nat
      hb : GE.ge b (HAdd.hAdd N k)
      ⊢ Membership.mem V { fst := u a, snd := u b }
    -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    convert h (a - k) (Nat.le_sub_of_add_le ha) (b - k) (Nat.le_sub_of_add_le hb) <;> omega
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      u : Nat → α
      k : Nat
      h : CauchySeq u
      ⊢ CauchySeq fun n => u (HAdd.hAdd n k)
    -/
  · exact h.comp_tendsto (tendsto_add_atTop_nat k)
    /-
      🎉 no goals
    -/


theorem Filter.HasBasis.cauchySeq_iff {γ} [Nonempty β] [SemilatticeSup β] {u : β → α} {p : γ → Prop}
    {s : γ → Set (α × α)} (h : (𝓤 α).HasBasis p s) :
    CauchySeq u ↔ ∀ i, p i → ∃ N, ∀ m, N ≤ m → ∀ n, N ≤ n → (u m, u n) ∈ s i := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    γ : Sort u_1
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    u : β → α
    p : γ → Prop
    s : γ → Set (Prod α α)
    h : (uniformity α).HasBasis p s
    ⊢ Iff (CauchySeq u) (∀ (i : γ), p i → Exists fun N => ∀ (m : β), LE.le N m → ∀ …
  -/
  rw [cauchySeq_iff_tendsto, ← prod_atTop_atTop_eq]
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    γ : Sort u_1
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    u : β → α
    p : γ → Prop
    s : γ → Set (Prod α α)
    h : (uniformity α).HasBasis p s
    ⊢ Iff (Filter.Tendsto (Prod.map u u) (SProd.sprod Filter.atTop Filter.atTop) ( …
  -/
  refine (atTop_basis.prod_self.tendsto_iff h).trans ?_
  simp only [exists_prop, true_and, MapsTo, preimage, subset_def, Prod.forall, mem_prod_eq,
    mem_setOf_eq, mem_Ici, and_imp, Prod.map, @forall_swap (_ ≤ _) β]


theorem Filter.HasBasis.cauchySeq_iff' {γ} [Nonempty β] [SemilatticeSup β] {u : β → α}
    {p : γ → Prop} {s : γ → Set (α × α)} (H : (𝓤 α).HasBasis p s) :
    CauchySeq u ↔ ∀ i, p i → ∃ N, ∀ n ≥ N, (u n, u N) ∈ s i := by
  /-
    α : Type u
    β : Type v
    uniformSpace : UniformSpace α
    γ : Sort u_1
    inst✝¹ : Nonempty β
    inst✝ : SemilatticeSup β
    u : β → α
    p : γ → Prop
    s : γ → Set (Prod α α)
    H : (uniformity α).HasBasis p s
    ⊢ Iff (CauchySeq u) (∀ (i : γ), p i → Exists fun N => ∀ (n : β), GE.ge n N → M …
  -/
  refine H.cauchySeq_iff.trans ⟨fun h i hi => ?_, fun h i hi => ?_⟩
    /-
      case refine_1
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      γ : Sort u_1
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      p : γ → Prop
      s : γ → Set (Prod α α)
      H : (uniformity α).HasBasis p s
      h : ∀ (i : γ), p i → Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N …
      i : γ
      hi : p i
      ⊢ Exists fun N => ∀ (n : β), GE.ge n N → Membership.mem (s i) { fst := u n, sn …
    -/
  · exact (h i hi).imp fun N hN n hn => hN n hn N le_rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      γ : Sort u_1
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      p : γ → Prop
      s : γ → Set (Prod α α)
      H : (uniformity α).HasBasis p s
      h : ∀ (i : γ), p i → Exists fun N => ∀ (n : β), GE.ge n N → Membership.mem (s  …
      i : γ
      hi : p i
      ⊢ Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → Membership.mem …
    -/
  · rcases comp_symm_of_uniformity (H.mem_of_mem hi) with ⟨t, ht, ht', hts⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      γ : Sort u_1
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      p : γ → Prop
      s : γ → Set (Prod α α)
      H : (uniformity α).HasBasis p s
      h : ∀ (i : γ), p i → Exists fun N => ∀ (n : β), GE.ge n N → Membership.mem (s  …
      i : γ
      hi : p i
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      ht' : ∀ {a b : α}, Membership.mem t { fst := a, snd := b } → Membership.mem t  …
      hts : HasSubset.Subset (compRel t t) (s i)
      ⊢ Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → Membership.mem …
    -/
    rcases H.mem_iff.1 ht with ⟨j, hj, hjt⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      γ : Sort u_1
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      p : γ → Prop
      s : γ → Set (Prod α α)
      H : (uniformity α).HasBasis p s
      h : ∀ (i : γ), p i → Exists fun N => ∀ (n : β), GE.ge n N → Membership.mem (s  …
      i : γ
      hi : p i
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      ht' : ∀ {a b : α}, Membership.mem t { fst := a, snd := b } → Membership.mem t  …
      hts : HasSubset.Subset (compRel t t) (s i)
      j : γ
      hj : p j
      hjt : HasSubset.Subset (s j) t
      ⊢ Exists fun N => ∀ (m : β), LE.le N m → ∀ (n : β), LE.le N n → Membership.mem …
    -/
    refine (h j hj).imp fun N hN m hm n hn => hts ⟨u N, hjt ?_, ht' <| hjt ?_⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.refine_1
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      γ : Sort u_1
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      p : γ → Prop
      s : γ → Set (Prod α α)
      H : (uniformity α).HasBasis p s
      h : ∀ (i : γ), p i → Exists fun N => ∀ (n : β), GE.ge n N → Membership.mem (s  …
      i : γ
      hi : p i
      t : Set (Prod α α)
      ht : Membership.mem (uniformity α) t
      ht' : ∀ {a b : α}, Membership.mem t { fst := a, snd := b } → Membership.mem t  …
      hts : HasSubset.Subset (compRel t t) (s i)
      j : γ
      hj : p j
      hjt : HasSubset.Subset (s j) t
      N : β
      hN : ∀ (n : β), GE.ge n N → Membership.mem (s j) { fst := u n, snd := u N }
      m : β
      hm : LE.le N m
      n : β
      hn : LE.le N n
      ⊢ Membership.mem (s j) { fst := { fst := u m, snd := u n }.1, snd := u N }
    -/
    exacts [hN m hm, hN n hn]
    /-
      🎉 no goals
    -/


theorem cauchySeq_of_controlled [SemilatticeSup β] [Nonempty β] (U : β → Set (α × α))
    (hU : ∀ s ∈ 𝓤 α, ∃ n, U n ⊆ s) {f : β → α}
    (hf : ∀ ⦃N m n : β⦄, N ≤ m → N ≤ n → (f m, f n) ∈ U N) : CauchySeq f :=
    -- Porting note: changed to semi-implicit arguments
  cauchySeq_iff_tendsto.2
    (by
      /-
        α : Type u
        β : Type v
        uniformSpace : UniformSpace α
        inst✝¹ : SemilatticeSup β
        inst✝ : Nonempty β
        U : β → Set (Prod α α)
        hU : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n => …
        f : β → α
        hf : ∀ ⦃N m n : β⦄, LE.le N m → LE.le N n → Membership.mem (U N) { fst := f m, …
        ⊢ Filter.Tendsto (Prod.map f f) Filter.atTop (uniformity α)
      -/
      intro s hs
      /-
        α : Type u
        β : Type v
        uniformSpace : UniformSpace α
        inst✝¹ : SemilatticeSup β
        inst✝ : Nonempty β
        U : β → Set (Prod α α)
        hU : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n => …
        f : β → α
        hf : ∀ ⦃N m n : β⦄, LE.le N m → LE.le N n → Membership.mem (U N) { fst := f m, …
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        ⊢ Membership.mem (Filter.map (Prod.map f f) Filter.atTop) s
      -/
      rw [mem_map, mem_atTop_sets]
      /-
        α : Type u
        β : Type v
        uniformSpace : UniformSpace α
        inst✝¹ : SemilatticeSup β
        inst✝ : Nonempty β
        U : β → Set (Prod α α)
        hU : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n => …
        f : β → α
        hf : ∀ ⦃N m n : β⦄, LE.le N m → LE.le N n → Membership.mem (U N) { fst := f m, …
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        ⊢ Exists fun a => ∀ (b : Prod β β), GE.ge b a → Membership.mem (Set.preimage ( …
      -/
      cases' hU s hs with N hN
      /-
        case intro
        α : Type u
        β : Type v
        uniformSpace : UniformSpace α
        inst✝¹ : SemilatticeSup β
        inst✝ : Nonempty β
        U : β → Set (Prod α α)
        hU : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n => …
        f : β → α
        hf : ∀ ⦃N m n : β⦄, LE.le N m → LE.le N n → Membership.mem (U N) { fst := f m, …
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        N : β
        hN : HasSubset.Subset (U N) s
        ⊢ Exists fun a => ∀ (b : Prod β β), GE.ge b a → Membership.mem (Set.preimage ( …
      -/
      refine ⟨(N, N), fun mn hmn => ?_⟩
      /-
        case intro
        α : Type u
        β : Type v
        uniformSpace : UniformSpace α
        inst✝¹ : SemilatticeSup β
        inst✝ : Nonempty β
        U : β → Set (Prod α α)
        hU : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n => …
        f : β → α
        hf : ∀ ⦃N m n : β⦄, LE.le N m → LE.le N n → Membership.mem (U N) { fst := f m, …
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        N : β
        hN : HasSubset.Subset (U N) s
        mn : Prod β β
        hmn : GE.ge mn { fst := N, snd := N }
        ⊢ Membership.mem (Set.preimage (Prod.map f f) s) mn
      -/
      cases' mn with m n
      /-
        case intro.mk
        α : Type u
        β : Type v
        uniformSpace : UniformSpace α
        inst✝¹ : SemilatticeSup β
        inst✝ : Nonempty β
        U : β → Set (Prod α α)
        hU : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n => …
        f : β → α
        hf : ∀ ⦃N m n : β⦄, LE.le N m → LE.le N n → Membership.mem (U N) { fst := f m, …
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        N : β
        hN : HasSubset.Subset (U N) s
        m n : β
        hmn : GE.ge { fst := m, snd := n } { fst := N, snd := N }
        ⊢ Membership.mem (Set.preimage (Prod.map f f) s) { fst := m, snd := n }
      -/
      exact hN (hf hmn.1 hmn.2))
      /-
        🎉 no goals
      -/


theorem isComplete_iff_clusterPt {s : Set α} :
    IsComplete s ↔ ∀ l, Cauchy l → l ≤ 𝓟 s → ∃ x ∈ s, ClusterPt x l :=
  forall₃_congr fun _ hl _ => exists_congr fun _ => and_congr_right fun _ =>
    le_nhds_iff_adhp_of_cauchy hl


theorem isComplete_iff_ultrafilter {s : Set α} :
    IsComplete s ↔ ∀ l : Ultrafilter α, Cauchy (l : Filter α) → ↑l ≤ 𝓟 s → ∃ x ∈ s, ↑l ≤ 𝓝 x := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    ⊢ Iff (IsComplete s) (∀ (l : Ultrafilter α), Cauchy ↑l → LE.le (↑l) (Filter.pr …
  -/
  refine ⟨fun h l => h l, fun H => isComplete_iff_clusterPt.2 fun l hl hls => ?_⟩
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    H : ∀ (l : Ultrafilter α), Cauchy ↑l → LE.le (↑l) (Filter.principal s) → Exist …
    l : Filter α
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    ⊢ Exists fun x => And (Membership.mem s x) (ClusterPt x l)
  -/
  haveI := hl.1
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    H : ∀ (l : Ultrafilter α), Cauchy ↑l → LE.le (↑l) (Filter.principal s) → Exist …
    l : Filter α
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    ⊢ Exists fun x => And (Membership.mem s x) (ClusterPt x l)
  -/
  rcases H (Ultrafilter.of l) hl.ultrafilter_of ((Ultrafilter.of_le l).trans hls) with ⟨x, hxs, hxl⟩
  /-
    case intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    H : ∀ (l : Ultrafilter α), Cauchy ↑l → LE.le (↑l) (Filter.principal s) → Exist …
    l : Filter α
    hl : Cauchy l
    hls : LE.le l (Filter.principal s)
    this : l.NeBot
    x : α
    hxs : Membership.mem s x
    hxl : LE.le (↑(Ultrafilter.of l)) (nhds x)
    ⊢ Exists fun x => And (Membership.mem s x) (ClusterPt x l)
  -/
  exact ⟨x, hxs, (ClusterPt.of_le_nhds hxl).mono (Ultrafilter.of_le l)⟩
  /-
    🎉 no goals
  -/


theorem isComplete_iff_ultrafilter' {s : Set α} :
    IsComplete s ↔ ∀ l : Ultrafilter α, Cauchy (l : Filter α) → s ∈ l → ∃ x ∈ s, ↑l ≤ 𝓝 x :=
                                         /-
                                           α : Type u
                                           uniformSpace : UniformSpace α
                                           s : Set α
                                           ⊢ Iff (∀ (l : Ultrafilter α), Cauchy ↑l → LE.le (↑l) (Filter.principal s) → Ex …
                                         -/
  isComplete_iff_ultrafilter.trans <| by simp only [le_principal_iff, Ultrafilter.mem_coe]
                                         /-
                                           🎉 no goals
                                         -/


protected theorem IsComplete.union {s t : Set α} (hs : IsComplete s) (ht : IsComplete t) :
    IsComplete (s ∪ t) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s t : Set α
    hs : IsComplete s
    ht : IsComplete t
    ⊢ IsComplete (Union.union s t)
  -/
  simp only [isComplete_iff_ultrafilter', Ultrafilter.union_mem_iff, or_imp] at *
  exact fun l hl =>
    ⟨fun hsl => (hs l hl hsl).imp fun x hx => ⟨Or.inl hx.1, hx.2⟩, fun htl =>
      (ht l hl htl).imp fun x hx => ⟨Or.inr hx.1, hx.2⟩⟩


theorem isComplete_iUnion_separated {ι : Sort*} {s : ι → Set α} (hs : ∀ i, IsComplete (s i))
    {U : Set (α × α)} (hU : U ∈ 𝓤 α) (hd : ∀ (i j : ι), ∀ x ∈ s i, ∀ y ∈ s j, (x, y) ∈ U → i = j) :
    IsComplete (⋃ i, s i) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    s : ι → Set α
    hs : ∀ (i : ι), IsComplete (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hd : ∀ (i j : ι) (x : α), Membership.mem (s i) x → ∀ (y : α), Membership.mem ( …
    ⊢ IsComplete (Set.iUnion fun i => s i)
  -/
  set S := ⋃ i, s i
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    s : ι → Set α
    hs : ∀ (i : ι), IsComplete (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hd : ∀ (i j : ι) (x : α), Membership.mem (s i) x → ∀ (y : α), Membership.mem ( …
    S : Set α := Set.iUnion fun i => s i
    ⊢ IsComplete S
  -/
  intro l hl hls
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    s : ι → Set α
    hs : ∀ (i : ι), IsComplete (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hd : ∀ (i j : ι) (x : α), Membership.mem (s i) x → ∀ (y : α), Membership.mem ( …
    S : Set α := Set.iUnion fun i => s i
    l : Filter α
    hl : Cauchy l
    hls : LE.le l (Filter.principal S)
    ⊢ Exists fun x => And (Membership.mem S x) (LE.le l (nhds x))
  -/
  rw [le_principal_iff] at hls
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    s : ι → Set α
    hs : ∀ (i : ι), IsComplete (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hd : ∀ (i j : ι) (x : α), Membership.mem (s i) x → ∀ (y : α), Membership.mem ( …
    S : Set α := Set.iUnion fun i => s i
    l : Filter α
    hl : Cauchy l
    hls : Membership.mem l S
    ⊢ Exists fun x => And (Membership.mem S x) (LE.le l (nhds x))
  -/
  cases' cauchy_iff.1 hl with hl_ne hl'
  obtain ⟨t, htS, htl, htU⟩ : ∃ t, t ⊆ S ∧ t ∈ l ∧ t ×ˢ t ⊆ U := by
    rcases hl' U hU with ⟨t, htl, htU⟩
    refine ⟨t ∩ S, inter_subset_right, inter_mem htl hls, Subset.trans ?_ htU⟩
    gcongr <;> apply inter_subset_left
  obtain ⟨i, hi⟩ : ∃ i, t ⊆ s i := by
    rcases Filter.nonempty_of_mem htl with ⟨x, hx⟩
    rcases mem_iUnion.1 (htS hx) with ⟨i, hi⟩
    refine ⟨i, fun y hy => ?_⟩
    rcases mem_iUnion.1 (htS hy) with ⟨j, hj⟩
    rwa [hd i j x hi y hj (htU <| mk_mem_prod hx hy)]
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    s : ι → Set α
    hs : ∀ (i : ι), IsComplete (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hd : ∀ (i j : ι) (x : α), Membership.mem (s i) x → ∀ (y : α), Membership.mem ( …
    S : Set α := Set.iUnion fun i => s i
    l : Filter α
    hl : Cauchy l
    hls : Membership.mem l S
    hl_ne : l.NeBot
    hl' : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t = …
    t : Set α
    htS : HasSubset.Subset t S
    htl : Membership.mem l t
    htU : HasSubset.Subset (SProd.sprod t t) U
    i : ι
    hi : HasSubset.Subset t (s i)
    ⊢ Exists fun x => And (Membership.mem S x) (LE.le l (nhds x))
  -/
  rcases hs i l hl (le_principal_iff.2 <| mem_of_superset htl hi) with ⟨x, hxs, hlx⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    s : ι → Set α
    hs : ∀ (i : ι), IsComplete (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    hd : ∀ (i j : ι) (x : α), Membership.mem (s i) x → ∀ (y : α), Membership.mem ( …
    S : Set α := Set.iUnion fun i => s i
    l : Filter α
    hl : Cauchy l
    hls : Membership.mem l S
    hl_ne : l.NeBot
    hl' : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun t = …
    t : Set α
    htS : HasSubset.Subset t S
    htl : Membership.mem l t
    htU : HasSubset.Subset (SProd.sprod t t) U
    i : ι
    hi : HasSubset.Subset t (s i)
    x : α
    hxs : Membership.mem (s i) x
    hlx : LE.le l (nhds x)
    ⊢ Exists fun x => And (Membership.mem S x) (LE.le l (nhds x))
  -/
  exact ⟨x, mem_iUnion.2 ⟨i, hxs⟩, hlx⟩
  /-
    🎉 no goals
  -/


/-- A complete space is defined here using uniformities. A uniform space
  is complete if every Cauchy filter converges. -/
class CompleteSpace (α : Type u) [UniformSpace α] : Prop where
  /-- In a complete uniform space, every Cauchy filter converges. -/
  complete : ∀ {f : Filter α}, Cauchy f → ∃ x, f ≤ 𝓝 x


theorem complete_univ {α : Type u} [UniformSpace α] [CompleteSpace α] :
    IsComplete (univ : Set α) := fun f hf _ => by
  /-
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : CompleteSpace α
    f : Filter α
    hf : Cauchy f
    x✝ : LE.le f (Filter.principal Set.univ)
    ⊢ Exists fun x => And (Membership.mem Set.univ x) (LE.le f (nhds x))
  -/
  rcases CompleteSpace.complete hf with ⟨x, hx⟩
  /-
    case intro
    α : Type u
    inst✝¹ : UniformSpace α
    inst✝ : CompleteSpace α
    f : Filter α
    hf : Cauchy f
    x✝ : LE.le f (Filter.principal Set.univ)
    x : α
    hx : LE.le f (nhds x)
    ⊢ Exists fun x => And (Membership.mem Set.univ x) (LE.le f (nhds x))
  -/
  exact ⟨x, mem_univ x, hx⟩
  /-
    🎉 no goals
  -/


instance CompleteSpace.prod [UniformSpace β] [CompleteSpace α] [CompleteSpace β] :
    CompleteSpace (α × β) where
  complete hf :=
    let ⟨x1, hx1⟩ := CompleteSpace.complete <| hf.map uniformContinuous_fst
    let ⟨x2, hx2⟩ := CompleteSpace.complete <| hf.map uniformContinuous_snd
                  /-
                    α : Type u
                    β : Type v
                    uniformSpace : UniformSpace α
                    inst✝² : UniformSpace β
                    inst✝¹ : CompleteSpace α
                    inst✝ : CompleteSpace β
                    f✝ : Filter (Prod α β)
                    hf : Cauchy f✝
                    x1 : α
                    hx1 : LE.le (Filter.map (fun p => p.1) f✝) (nhds x1)
                    x2 : β
                    hx2 : LE.le (Filter.map (fun p => p.2) f✝) (nhds x2)
                    ⊢ LE.le f✝ (nhds { fst := x1, snd := x2 })
                  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
    ⟨(x1, x2), by rw [nhds_prod_eq, le_prod]; constructor <;> assumption⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma CompleteSpace.fst_of_prod [UniformSpace β] [CompleteSpace (α × β)] [h : Nonempty β] :
    CompleteSpace α where
  complete hf :=
    let ⟨y⟩ := h
    let ⟨(a, b), hab⟩ := CompleteSpace.complete <| hf.prod <| cauchy_pure (a := y)
           /-
             α : Type u
             β : Type v
             uniformSpace : UniformSpace α
             inst✝¹ : UniformSpace β
             inst✝ : CompleteSpace (Prod α β)
             h : Nonempty β
             f✝ : Filter α
             hf : Cauchy f✝
             y : β
             a : α
             b : β
             hab : LE.le (SProd.sprod f✝ (Pure.pure y)) (nhds { fst := a, snd := b })
             ⊢ LE.le f✝ (nhds a)
           -/
    ⟨a, by simpa only [map_fst_prod, nhds_prod_eq] using map_mono (m := Prod.fst) hab⟩
           /-
             🎉 no goals
           -/


lemma CompleteSpace.snd_of_prod [UniformSpace β] [CompleteSpace (α × β)] [h : Nonempty α] :
    CompleteSpace β where
  complete hf :=
    let ⟨x⟩ := h
    let ⟨(a, b), hab⟩ := CompleteSpace.complete <| (cauchy_pure (a := x)).prod hf
           /-
             α : Type u
             β : Type v
             uniformSpace : UniformSpace α
             inst✝¹ : UniformSpace β
             inst✝ : CompleteSpace (Prod α β)
             h : Nonempty α
             f✝ : Filter β
             hf : Cauchy f✝
             x a : α
             b : β
             hab : LE.le (SProd.sprod (Pure.pure x) f✝) (nhds { fst := a, snd := b })
             ⊢ LE.le f✝ (nhds b)
           -/
    ⟨b, by simpa only [map_snd_prod, nhds_prod_eq] using map_mono (m := Prod.snd) hab⟩
           /-
             🎉 no goals
           -/


lemma completeSpace_prod_of_nonempty [UniformSpace β] [Nonempty α] [Nonempty β] :
    CompleteSpace (α × β) ↔ CompleteSpace α ∧ CompleteSpace β :=
  ⟨fun _ ↦ ⟨.fst_of_prod (β := β), .snd_of_prod (α := α)⟩, fun ⟨_, _⟩ ↦ .prod⟩


@[to_additive]
instance CompleteSpace.mulOpposite [CompleteSpace α] : CompleteSpace αᵐᵒᵖ where
  complete hf :=
    MulOpposite.op_surjective.exists.mpr <|
      let ⟨x, hx⟩ := CompleteSpace.complete (hf.map MulOpposite.uniformContinuous_unop)
      ⟨x, (map_le_iff_le_comap.mp hx).trans_eq <| MulOpposite.comap_unop_nhds _⟩


/-- If `univ` is complete, the space is a complete space -/
theorem completeSpace_of_isComplete_univ (h : IsComplete (univ : Set α)) : CompleteSpace α :=
  ⟨fun hf => let ⟨x, _, hx⟩ := h _ hf ((@principal_univ α).symm ▸ le_top); ⟨x, hx⟩⟩


theorem completeSpace_iff_isComplete_univ : CompleteSpace α ↔ IsComplete (univ : Set α) :=
  ⟨@complete_univ α _, completeSpace_of_isComplete_univ⟩


theorem completeSpace_iff_ultrafilter :
    CompleteSpace α ↔ ∀ l : Ultrafilter α, Cauchy (l : Filter α) → ∃ x : α, ↑l ≤ 𝓝 x := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ⊢ Iff (CompleteSpace α) (∀ (l : Ultrafilter α), Cauchy ↑l → Exists fun x => LE …
  -/
  simp [completeSpace_iff_isComplete_univ, isComplete_iff_ultrafilter]
  /-
    🎉 no goals
  -/


theorem cauchy_iff_exists_le_nhds [CompleteSpace α] {l : Filter α} [NeBot l] :
    Cauchy l ↔ ∃ x, l ≤ 𝓝 x :=
  ⟨CompleteSpace.complete, fun ⟨_, hx⟩ => cauchy_nhds.mono hx⟩


theorem cauchy_map_iff_exists_tendsto [CompleteSpace α] {l : Filter β} {f : β → α} [NeBot l] :
    Cauchy (l.map f) ↔ ∃ x, Tendsto f l (𝓝 x) :=
  cauchy_iff_exists_le_nhds


/-- A Cauchy sequence in a complete space converges -/
theorem cauchySeq_tendsto_of_complete [Preorder β] [CompleteSpace α] {u : β → α}
    (H : CauchySeq u) : ∃ x, Tendsto u atTop (𝓝 x) :=
  CompleteSpace.complete H


/-- If `K` is a complete subset, then any cauchy sequence in `K` converges to a point in `K` -/
theorem cauchySeq_tendsto_of_isComplete [Preorder β] {K : Set α} (h₁ : IsComplete K)
    {u : β → α} (h₂ : ∀ n, u n ∈ K) (h₃ : CauchySeq u) : ∃ v ∈ K, Tendsto u atTop (𝓝 v) :=
  h₁ _ h₃ <| le_principal_iff.2 <| mem_map_iff_exists_image.2
                        /-
                          α : Type u
                          β : Type v
                          uniformSpace : UniformSpace α
                          inst✝ : Preorder β
                          K : Set α
                          h₁ : IsComplete K
                          u : β → α
                          h₂ : ∀ (n : β), Membership.mem K (u n)
                          h₃ : CauchySeq u
                          ⊢ HasSubset.Subset (Set.image u Set.univ) K
                        -/
    ⟨univ, univ_mem, by rwa [image_univ, range_subset_iff]⟩
                        /-
                          🎉 no goals
                        -/


theorem Cauchy.le_nhds_lim [CompleteSpace α] {f : Filter α} (hf : Cauchy f) :
    haveI := hf.1.nonempty; f ≤ 𝓝 (lim f) :=
  _root_.le_nhds_lim (CompleteSpace.complete hf)


theorem CauchySeq.tendsto_limUnder [Preorder β] [CompleteSpace α] {u : β → α} (h : CauchySeq u) :
    haveI := h.1.nonempty; Tendsto u atTop (𝓝 <| limUnder atTop u) :=
  h.le_nhds_lim


theorem IsClosed.isComplete [CompleteSpace α] {s : Set α} (h : IsClosed s) : IsComplete s :=
  fun _ cf fs =>
  let ⟨x, hx⟩ := CompleteSpace.complete cf
  ⟨x, isClosed_iff_clusterPt.mp h x (cf.left.mono (le_inf hx fs)), hx⟩


/-- A set `s` is totally bounded if for every entourage `d` there is a finite
  set of points `t` such that every element of `s` is `d`-near to some element of `t`. -/
def TotallyBounded (s : Set α) : Prop :=
  ∀ d ∈ 𝓤 α, ∃ t : Set α, t.Finite ∧ s ⊆ ⋃ y ∈ t, { x | (x, y) ∈ d }


theorem TotallyBounded.exists_subset {s : Set α} (hs : TotallyBounded s) {U : Set (α × α)}
    (hU : U ∈ 𝓤 α) : ∃ t, t ⊆ s ∧ Set.Finite t ∧ s ⊆ ⋃ y ∈ t, { x | (x, y) ∈ U } := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    hs : TotallyBounded s
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  rcases comp_symm_of_uniformity hU with ⟨r, hr, rs, rU⟩
  /-
    case intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    hs : TotallyBounded s
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    r : Set (Prod α α)
    hr : Membership.mem (uniformity α) r
    rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
    rU : HasSubset.Subset (compRel r r) U
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  rcases hs r hr with ⟨k, fk, ks⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    hs : TotallyBounded s
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    r : Set (Prod α α)
    hr : Membership.mem (uniformity α) r
    rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
    rU : HasSubset.Subset (compRel r r) U
    k : Set α
    fk : k.Finite
    ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  let u := k ∩ { y | ∃ x ∈ s, (x, y) ∈ r }
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    hs : TotallyBounded s
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    r : Set (Prod α α)
    hr : Membership.mem (uniformity α) r
    rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
    rU : HasSubset.Subset (compRel r r) U
    k : Set α
    fk : k.Finite
    ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
    u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  choose f hfs hfr using fun x : u => x.coe_prop.2
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    hs : TotallyBounded s
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    r : Set (Prod α α)
    hr : Membership.mem (uniformity α) r
    rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
    rU : HasSubset.Subset (compRel r r) U
    k : Set α
    fk : k.Finite
    ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
    u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
    f : ↑u → α
    hfs : ∀ (x : ↑u), Membership.mem s (f x)
    hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (HasSubset.Subset s …
  -/
  refine ⟨range f, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      ⊢ HasSubset.Subset (Set.range f) s
    -/
  · exact range_subset_iff.2 hfs
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      ⊢ (Set.range f).Finite
    -/
  · haveI : Fintype u := (fk.inter_of_left _).fintype
    /-
      case intro.intro.intro.intro.intro.refine_2
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      this : Fintype ↑u
      ⊢ (Set.range f).Finite
    -/
    exact finite_range f
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_3
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      ⊢ HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x => M …
    -/
  · intro x xs
    /-
      case intro.intro.intro.intro.intro.refine_3
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      x : α
      xs : Membership.mem s x
      ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => setOf fun x => Membe …
    -/
    obtain ⟨y, hy, xy⟩ := mem_iUnion₂.1 (ks xs)
    /-
      case intro.intro.intro.intro.intro.refine_3.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      x : α
      xs : Membership.mem s x
      y : α
      hy : Membership.mem k y
      xy : Membership.mem (setOf fun x => Membership.mem r { fst := x, snd := y }) x
      ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => setOf fun x => Membe …
    -/
    rw [biUnion_range, mem_iUnion]
    /-
      case intro.intro.intro.intro.intro.refine_3.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      x : α
      xs : Membership.mem s x
      y : α
      hy : Membership.mem k y
      xy : Membership.mem (setOf fun x => Membership.mem r { fst := x, snd := y }) x
      ⊢ Exists fun i => Membership.mem (setOf fun x => Membership.mem U { fst := x,  …
    -/
    set z : ↥u := ⟨y, hy, ⟨x, xs, xy⟩⟩
    /-
      case intro.intro.intro.intro.intro.refine_3.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      hs : TotallyBounded s
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      r : Set (Prod α α)
      hr : Membership.mem (uniformity α) r
      rs : ∀ {a b : α}, Membership.mem r { fst := a, snd := b } → Membership.mem r { …
      rU : HasSubset.Subset (compRel r r) U
      k : Set α
      fk : k.Finite
      ks : HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x = …
      u : Set α := Inter.inter k (setOf fun y => Exists fun x => And (Membership.mem …
      f : ↑u → α
      hfs : ∀ (x : ↑u), Membership.mem s (f x)
      hfr : ∀ (x : ↑u), Membership.mem r { fst := f x, snd := ↑x }
      x : α
      xs : Membership.mem s x
      y : α
      hy : Membership.mem k y
      xy : Membership.mem (setOf fun x => Membership.mem r { fst := x, snd := y }) x
      z : ↑u := ⟨y, ⋯⟩
      ⊢ Exists fun i => Membership.mem (setOf fun x => Membership.mem U { fst := x,  …
    -/
    exact ⟨z, rU <| mem_compRel.2 ⟨y, xy, rs (hfr z)⟩⟩
    /-
      🎉 no goals
    -/


theorem totallyBounded_iff_subset {s : Set α} :
    TotallyBounded s ↔
      ∀ d ∈ 𝓤 α, ∃ t, t ⊆ s ∧ Set.Finite t ∧ s ⊆ ⋃ y ∈ t, { x | (x, y) ∈ d } :=
  ⟨fun H _ hd ↦ H.exists_subset hd, fun H d hd ↦ let ⟨t, _, ht⟩ := H d hd; ⟨t, ht⟩⟩


theorem Filter.HasBasis.totallyBounded_iff {ι} {p : ι → Prop} {U : ι → Set (α × α)}
    (H : (𝓤 α).HasBasis p U) {s : Set α} :
    TotallyBounded s ↔ ∀ i, p i → ∃ t : Set α, Set.Finite t ∧ s ⊆ ⋃ y ∈ t, { x | (x, y) ∈ U i } :=
  H.forall_iff fun _ _ hUV h =>
    h.imp fun _ ht => ⟨ht.1, ht.2.trans <| iUnion₂_mono fun _ _ _ hy => hUV hy⟩


theorem totallyBounded_of_forall_symm {s : Set α}
    (h : ∀ V ∈ 𝓤 α, SymmetricRel V → ∃ t : Set α, Set.Finite t ∧ s ⊆ ⋃ y ∈ t, ball y V) :
    TotallyBounded s :=
  UniformSpace.hasBasis_symmetric.totallyBounded_iff.2 fun V hV => by
    /-
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      h : ∀ (V : Set (Prod α α)), Membership.mem (uniformity α) V → SymmetricRel V → …
      V : Set (Prod α α)
      hV : And (Membership.mem (uniformity α) V) (SymmetricRel V)
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
    -/
    simpa only [ball_eq_of_symmetry hV.2] using h V hV.1 hV.2
    /-
      🎉 no goals
    -/


theorem TotallyBounded.subset {s₁ s₂ : Set α} (hs : s₁ ⊆ s₂) (h : TotallyBounded s₂) :
    TotallyBounded s₁ := fun d hd =>
  let ⟨t, ht₁, ht₂⟩ := h d hd
  ⟨t, ht₁, Subset.trans hs ht₂⟩


@[deprecated (since := "2024-06-01")]
alias totallyBounded_subset := TotallyBounded.subset


/-- The closure of a totally bounded set is totally bounded. -/
theorem TotallyBounded.closure {s : Set α} (h : TotallyBounded s) : TotallyBounded (closure s) :=
  uniformity_hasBasis_closed.totallyBounded_iff.2 fun V hV =>
    let ⟨t, htf, hst⟩ := h V hV.1
    ⟨t, htf,
      closure_minimal hst <|
        htf.isClosed_biUnion fun _ _ => hV.2.preimage (continuous_id.prod_mk continuous_const)⟩


@[simp]
lemma totallyBounded_closure {s : Set α} : TotallyBounded (closure s) ↔ TotallyBounded s :=
  ⟨fun h ↦ h.subset subset_closure, TotallyBounded.closure⟩


/-- A finite indexed union is totally bounded
if and only if each set of the family is totally bounded. -/
@[simp]
lemma totallyBounded_iUnion {ι : Sort*} [Finite ι] {s : ι → Set α} :
    TotallyBounded (⋃ i, s i) ↔ ∀ i, TotallyBounded (s i) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    inst✝ : Finite ι
    s : ι → Set α
    ⊢ Iff (TotallyBounded (Set.iUnion fun i => s i)) (∀ (i : ι), TotallyBounded (s …
  -/
  refine ⟨fun h i ↦ h.subset (subset_iUnion _ _), fun h U hU ↦ ?_⟩
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    inst✝ : Finite ι
    s : ι → Set α
    h : ∀ (i : ι), TotallyBounded (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.iUnion fun i => s i) (Se …
  -/
  choose t htf ht using (h · U hU)
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    inst✝ : Finite ι
    s : ι → Set α
    h : ∀ (i : ι), TotallyBounded (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    t : ι → Set α
    htf : ∀ (x : ι), (t x).Finite
    ht : ∀ (x : ι), HasSubset.Subset (s x) (Set.iUnion fun y => Set.iUnion fun h = …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.iUnion fun i => s i) (Se …
  -/
  refine ⟨⋃ i, t i, finite_iUnion htf, ?_⟩
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    inst✝ : Finite ι
    s : ι → Set α
    h : ∀ (i : ι), TotallyBounded (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    t : ι → Set α
    htf : ∀ (x : ι), (t x).Finite
    ht : ∀ (x : ι), HasSubset.Subset (s x) (Set.iUnion fun y => Set.iUnion fun h = …
    ⊢ HasSubset.Subset (Set.iUnion fun i => s i) (Set.iUnion fun y => Set.iUnion f …
  -/
  rw [biUnion_iUnion]
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Sort u_1
    inst✝ : Finite ι
    s : ι → Set α
    h : ∀ (i : ι), TotallyBounded (s i)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    t : ι → Set α
    htf : ∀ (x : ι), (t x).Finite
    ht : ∀ (x : ι), HasSubset.Subset (s x) (Set.iUnion fun y => Set.iUnion fun h = …
    ⊢ HasSubset.Subset (Set.iUnion fun i => s i) (Set.iUnion fun i => Set.iUnion f …
  -/
  gcongr; apply ht
          /-
            🎉 no goals
          -/


/-- A union indexed by a finite set is totally bounded
if and only if each set of the family is totally bounded. -/
lemma totallyBounded_biUnion {ι : Type*} {I : Set ι} (hI : I.Finite) {s : ι → Set α} :
    TotallyBounded (⋃ i ∈ I, s i) ↔ ∀ i ∈ I, TotallyBounded (s i) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Type u_1
    I : Set ι
    hI : I.Finite
    s : ι → Set α
    ⊢ Iff (TotallyBounded (Set.iUnion fun i => Set.iUnion fun h => s i)) (∀ (i : ι …
  -/
  have := hI.to_subtype
  /-
    α : Type u
    uniformSpace : UniformSpace α
    ι : Type u_1
    I : Set ι
    hI : I.Finite
    s : ι → Set α
    this : Finite ↑I
    ⊢ Iff (TotallyBounded (Set.iUnion fun i => Set.iUnion fun h => s i)) (∀ (i : ι …
  -/
  rw [biUnion_eq_iUnion, totallyBounded_iUnion, Subtype.forall]
  /-
    🎉 no goals
  -/


/-- A union of a finite family of sets is totally bounded
if and only if each set of the family is totally bounded. -/
lemma totallyBounded_sUnion {S : Set (Set α)} (hS : S.Finite) :
    TotallyBounded (⋃₀ S) ↔ ∀ s ∈ S, TotallyBounded s := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    S : Set (Set α)
    hS : S.Finite
    ⊢ Iff (TotallyBounded S.sUnion) (∀ (s : Set α), Membership.mem S s → TotallyBo …
  -/
  rw [sUnion_eq_biUnion, totallyBounded_biUnion hS]
  /-
    🎉 no goals
  -/


/-- A finite set is totally bounded. -/
lemma Set.Finite.totallyBounded {s : Set α} (hs : s.Finite) : TotallyBounded s := fun _U hU ↦
  ⟨s, hs, fun _x hx ↦ mem_biUnion hx <| refl_mem_uniformity hU⟩


/-- A subsingleton is totally bounded. -/
lemma Set.Subsingleton.totallyBounded {s : Set α} (hs : s.Subsingleton) :
    TotallyBounded s :=
  hs.finite.totallyBounded


@[simp]
lemma totallyBounded_singleton (a : α) : TotallyBounded {a} := (finite_singleton a).totallyBounded


@[simp]
theorem totallyBounded_empty : TotallyBounded (∅ : Set α) := finite_empty.totallyBounded


/-- The union of two sets is totally bounded
if and only if each of the two sets is totally bounded.-/
@[simp]
lemma totallyBounded_union {s t : Set α} :
    TotallyBounded (s ∪ t) ↔ TotallyBounded s ∧ TotallyBounded t := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s t : Set α
    ⊢ Iff (TotallyBounded (Union.union s t)) (And (TotallyBounded s) (TotallyBound …
  -/
  rw [union_eq_iUnion, totallyBounded_iUnion]
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s t : Set α
    ⊢ Iff (∀ (i : Bool), TotallyBounded (cond i s t)) (And (TotallyBounded s) (Tot …
  -/
  simp [and_comm]
  /-
    🎉 no goals
  -/


/-- The union of two totally bounded sets is totally bounded. -/
protected lemma TotallyBounded.union {s t : Set α} (hs : TotallyBounded s) (ht : TotallyBounded t) :
    TotallyBounded (s ∪ t) :=
  totallyBounded_union.2 ⟨hs, ht⟩


@[simp]
lemma totallyBounded_insert (a : α) {s : Set α} :
    TotallyBounded (insert a s) ↔ TotallyBounded s := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    a : α
    s : Set α
    ⊢ Iff (TotallyBounded (Insert.insert a s)) (TotallyBounded s)
  -/
  simp_rw [← singleton_union, totallyBounded_union, totallyBounded_singleton, true_and]
  /-
    🎉 no goals
  -/


protected alias ⟨_, TotallyBounded.insert⟩ := totallyBounded_insert


/-- The image of a totally bounded set under a uniformly continuous map is totally bounded. -/
theorem TotallyBounded.image [UniformSpace β] {f : α → β} {s : Set α} (hs : TotallyBounded s)
    (hf : UniformContinuous f) : TotallyBounded (f '' s) := fun t ht =>
  have : { p : α × α | (f p.1, f p.2) ∈ t } ∈ 𝓤 α := hf ht
  let ⟨c, hfc, hct⟩ := hs _ this
  ⟨f '' c, hfc.image f, by
    simp only [mem_image, iUnion_exists, biUnion_and', iUnion_iUnion_eq_right, image_subset_iff,
      preimage_iUnion, preimage_setOf_eq]
    simp? [subset_def] at hct says
      simp only [mem_setOf_eq, subset_def, mem_iUnion, exists_prop] at hct
    /-
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      s : Set α
      hs : TotallyBounded s
      hf : UniformContinuous f
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      this : Membership.mem (uniformity α) (setOf fun p => Membership.mem t { fst := …
      c : Set α
      hfc : c.Finite
      hct : ∀ (x : α), Membership.mem s x → Exists fun i => And (Membership.mem c i) …
      ⊢ HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun x => setOf fun a => M …
    -/
    intro x hx
    /-
      α : Type u
      β : Type v
      uniformSpace : UniformSpace α
      inst✝ : UniformSpace β
      f : α → β
      s : Set α
      hs : TotallyBounded s
      hf : UniformContinuous f
      t : Set (Prod β β)
      ht : Membership.mem (uniformity β) t
      this : Membership.mem (uniformity α) (setOf fun p => Membership.mem t { fst := …
      c : Set α
      hfc : c.Finite
      hct : ∀ (x : α), Membership.mem s x → Exists fun i => And (Membership.mem c i) …
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun x => setOf fun a => Membe …
    -/
    simpa using hct x hx⟩
    /-
      🎉 no goals
    -/


theorem Ultrafilter.cauchy_of_totallyBounded {s : Set α} (f : Ultrafilter α) (hs : TotallyBounded s)
    (h : ↑f ≤ 𝓟 s) : Cauchy (f : Filter α) :=
  ⟨f.neBot', fun _ ht =>
    let ⟨t', ht'₁, ht'_symm, ht'_t⟩ := comp_symm_of_uniformity ht
    let ⟨i, hi, hs_union⟩ := hs t' ht'₁
    have : (⋃ y ∈ i, { x | (x, y) ∈ t' }) ∈ f := mem_of_superset (le_principal_iff.mp h) hs_union
    have : ∃ y ∈ i, { x | (x, y) ∈ t' } ∈ f := (Ultrafilter.finite_biUnion_mem_iff hi).1 this
    let ⟨y, _, hif⟩ := this
    have : { x | (x, y) ∈ t' } ×ˢ { x | (x, y) ∈ t' } ⊆ compRel t' t' :=
      fun ⟨_, _⟩ ⟨(h₁ : (_, y) ∈ t'), (h₂ : (_, y) ∈ t')⟩ => ⟨y, h₁, ht'_symm h₂⟩
    mem_of_superset (prod_mem_prod hif hif) (Subset.trans this ht'_t)⟩


theorem totallyBounded_iff_filter {s : Set α} :
    TotallyBounded s ↔ ∀ f, NeBot f → f ≤ 𝓟 s → ∃ c ≤ f, Cauchy c := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    ⊢ Iff (TotallyBounded s) (∀ (f : Filter α), f.NeBot → LE.le f (Filter.principa …
  -/
  constructor
  · exact fun H f hf hfs => ⟨Ultrafilter.of f, Ultrafilter.of_le f,
      (Ultrafilter.of f).cauchy_of_totallyBounded H ((Ultrafilter.of_le f).trans hfs)⟩
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      ⊢ (∀ (f : Filter α), f.NeBot → LE.le f (Filter.principal s) → Exists fun c =>  …
    -/
  · intro H d hd
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      H : ∀ (f : Filter α), f.NeBot → LE.le f (Filter.principal s) → Exists fun c => …
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset s (Set.iUnion fun y => Set.iU …
    -/
    contrapose! H with hd_cover
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      ⊢ Exists fun f => And f.NeBot (And (LE.le f (Filter.principal s)) (∀ (c : Filt …
    -/
    set f := ⨅ t : Finset α, 𝓟 (s \ ⋃ y ∈ t, { x | (x, y) ∈ d })
    have hb : HasAntitoneBasis f fun t : Finset α ↦ s \ ⋃ y ∈ t, { x | (x, y) ∈ d } :=
      .iInf_principal fun _ _ ↦ diff_subset_diff_right ∘ biUnion_subset_biUnion_left
    have : Filter.NeBot f := hb.1.neBot_iff.2 fun _ ↦
      diff_nonempty.2 <| hd_cover _ (Finset.finite_toSet _)
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this : f.NeBot
      ⊢ Exists fun f => And f.NeBot (And (LE.le f (Filter.principal s)) (∀ (c : Filt …
    -/
    have : f ≤ 𝓟 s := iInf_le_of_le ∅ (by simp)
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this✝ : f.NeBot
      this : LE.le f (Filter.principal s)
      ⊢ Exists fun f => And f.NeBot (And (LE.le f (Filter.principal s)) (∀ (c : Filt …
    -/
    refine ⟨f, ‹_›, ‹_›, fun c hcf hc => ?_⟩
    /-
      case mpr
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this✝ : f.NeBot
      this : LE.le f (Filter.principal s)
      c : Filter α
      hcf : LE.le c f
      hc : Cauchy c
      ⊢ False
    -/
    rcases mem_prod_same_iff.1 (hc.2 hd) with ⟨m, hm, hmd⟩
    /-
      case mpr.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this✝ : f.NeBot
      this : LE.le f (Filter.principal s)
      c : Filter α
      hcf : LE.le c f
      hc : Cauchy c
      m : Set α
      hm : Membership.mem c m
      hmd : HasSubset.Subset (SProd.sprod m m) d
      ⊢ False
    -/
    rcases hc.1.nonempty_of_mem hm with ⟨y, hym⟩
    /-
      case mpr.intro.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this✝ : f.NeBot
      this : LE.le f (Filter.principal s)
      c : Filter α
      hcf : LE.le c f
      hc : Cauchy c
      m : Set α
      hm : Membership.mem c m
      hmd : HasSubset.Subset (SProd.sprod m m) d
      y : α
      hym : Membership.mem m y
      ⊢ False
    -/
    have : s \ {x | (x, y) ∈ d} ∈ c := by simpa using hcf (hb.mem {y})
    /-
      case mpr.intro.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this✝¹ : f.NeBot
      this✝ : LE.le f (Filter.principal s)
      c : Filter α
      hcf : LE.le c f
      hc : Cauchy c
      m : Set α
      hm : Membership.mem c m
      hmd : HasSubset.Subset (SProd.sprod m m) d
      y : α
      hym : Membership.mem m y
      this : Membership.mem c (SDiff.sdiff s (setOf fun x => Membership.mem d { fst  …
      ⊢ False
    -/
    rcases hc.1.nonempty_of_mem (inter_mem hm this) with ⟨z, hzm, -, hyz⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      s : Set α
      d : Set (Prod α α)
      hd : Membership.mem (uniformity α) d
      hd_cover : ∀ (t : Set α), t.Finite → Not (HasSubset.Subset s (Set.iUnion fun y …
      f : Filter α := iInf fun t => Filter.principal (SDiff.sdiff s (Set.iUnion fun  …
      hb : f.HasAntitoneBasis fun t => SDiff.sdiff s (Set.iUnion fun y => Set.iUnion …
      this✝¹ : f.NeBot
      this✝ : LE.le f (Filter.principal s)
      c : Filter α
      hcf : LE.le c f
      hc : Cauchy c
      m : Set α
      hm : Membership.mem c m
      hmd : HasSubset.Subset (SProd.sprod m m) d
      y : α
      hym : Membership.mem m y
      this : Membership.mem c (SDiff.sdiff s (setOf fun x => Membership.mem d { fst  …
      z : α
      hzm : Membership.mem m z
      hyz : Not (Membership.mem (setOf fun x => Membership.mem d { fst := x, snd :=  …
      ⊢ False
    -/
    exact hyz (hmd ⟨hzm, hym⟩)
    /-
      🎉 no goals
    -/


theorem totallyBounded_iff_ultrafilter {s : Set α} :
    TotallyBounded s ↔ ∀ f : Ultrafilter α, ↑f ≤ 𝓟 s → Cauchy (f : Filter α) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    ⊢ Iff (TotallyBounded s) (∀ (f : Ultrafilter α), LE.le (↑f) (Filter.principal  …
  -/
  refine ⟨fun hs f => f.cauchy_of_totallyBounded hs, fun H => totallyBounded_iff_filter.2 ?_⟩
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    H : ∀ (f : Ultrafilter α), LE.le (↑f) (Filter.principal s) → Cauchy ↑f
    ⊢ ∀ (f : Filter α), f.NeBot → LE.le f (Filter.principal s) → Exists fun c => A …
  -/
  intro f hf hfs
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Set α
    H : ∀ (f : Ultrafilter α), LE.le (↑f) (Filter.principal s) → Cauchy ↑f
    f : Filter α
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    ⊢ Exists fun c => And (LE.le c f) (Cauchy c)
  -/
  exact ⟨Ultrafilter.of f, Ultrafilter.of_le f, H _ ((Ultrafilter.of_le f).trans hfs)⟩
  /-
    🎉 no goals
  -/


theorem isCompact_iff_totallyBounded_isComplete {s : Set α} :
    IsCompact s ↔ TotallyBounded s ∧ IsComplete s :=
  ⟨fun hs =>
    ⟨totallyBounded_iff_ultrafilter.2 fun f hf =>
        let ⟨_, _, fx⟩ := isCompact_iff_ultrafilter_le_nhds.1 hs f hf
        cauchy_nhds.mono fx,
      fun f fc fs =>
      let ⟨a, as, fa⟩ := @hs f fc.1 fs
      ⟨a, as, le_nhds_of_cauchy_adhp fc fa⟩⟩,
    fun ⟨ht, hc⟩ =>
    isCompact_iff_ultrafilter_le_nhds.2 fun f hf =>
      hc _ (totallyBounded_iff_ultrafilter.1 ht f hf) hf⟩


protected theorem IsCompact.totallyBounded {s : Set α} (h : IsCompact s) : TotallyBounded s :=
  (isCompact_iff_totallyBounded_isComplete.1 h).1


protected theorem IsCompact.isComplete {s : Set α} (h : IsCompact s) : IsComplete s :=
  (isCompact_iff_totallyBounded_isComplete.1 h).2

-- see Note [lower instance priority]

instance (priority := 100) complete_of_compact {α : Type u} [UniformSpace α] [CompactSpace α] :
    CompleteSpace α :=
                /-
                  α✝ : Type u
                  β : Type v
                  uniformSpace : UniformSpace α✝
                  α : Type u
                  inst✝¹ : UniformSpace α
                  inst✝ : CompactSpace α
                  f✝ : Filter α
                  hf : Cauchy f✝
                  ⊢ Exists fun x => LE.le f✝ (nhds x)
                -/
  ⟨fun hf => by simpa using (isCompact_iff_totallyBounded_isComplete.1 isCompact_univ).2 _ hf⟩
                /-
                  🎉 no goals
                -/


theorem isCompact_of_totallyBounded_isClosed [CompleteSpace α] {s : Set α} (ht : TotallyBounded s)
    (hc : IsClosed s) : IsCompact s :=
  (@isCompact_iff_totallyBounded_isComplete α _ s).2 ⟨ht, hc.isComplete⟩


/-- Every Cauchy sequence over `ℕ` is totally bounded. -/
theorem CauchySeq.totallyBounded_range {s : ℕ → α} (hs : CauchySeq s) :
    TotallyBounded (range s) := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    ⊢ TotallyBounded (Set.range s)
  -/
  intro a ha
  /-
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.range s) (Set.iUnion fun …
  -/
  cases' cauchySeq_iff.1 hs a ha with n hn
  /-
    case intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    n : Nat
    hn : ∀ (k : Nat), GE.ge k n → ∀ (l : Nat), GE.ge l n → Membership.mem a { fst  …
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset (Set.range s) (Set.iUnion fun …
  -/
  refine ⟨s '' { k | k ≤ n }, (finite_le_nat _).image _, ?_⟩
  /-
    case intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    n : Nat
    hn : ∀ (k : Nat), GE.ge k n → ∀ (l : Nat), GE.ge l n → Membership.mem a { fst  …
    ⊢ HasSubset.Subset (Set.range s) (Set.iUnion fun y => Set.iUnion fun h => setO …
  -/
  rw [range_subset_iff, biUnion_image]
  /-
    case intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    n : Nat
    hn : ∀ (k : Nat), GE.ge k n → ∀ (l : Nat), GE.ge l n → Membership.mem a { fst  …
    ⊢ ∀ (y : Nat), Membership.mem (Set.iUnion fun y => Set.iUnion fun h => setOf f …
  -/
  intro m
  /-
    case intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    n : Nat
    hn : ∀ (k : Nat), GE.ge k n → ∀ (l : Nat), GE.ge l n → Membership.mem a { fst  …
    m : Nat
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => setOf fun x => Membe …
  -/
  rw [mem_iUnion₂]
  /-
    case intro
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    n : Nat
    hn : ∀ (k : Nat), GE.ge k n → ∀ (l : Nat), GE.ge l n → Membership.mem a { fst  …
    m : Nat
    ⊢ Exists fun i => Exists fun j => Membership.mem (setOf fun x => Membership.me …
  -/
  rcases le_total m n with hm | hm
  /-
    case intro.inl
    α : Type u
    uniformSpace : UniformSpace α
    s : Nat → α
    hs : CauchySeq s
    a : Set (Prod α α)
    ha : Membership.mem (uniformity α) a
    n : Nat
    hn : ∀ (k : Nat), GE.ge k n → ∀ (l : Nat), GE.ge l n → Membership.mem a { fst  …
    m : Nat
    hm : LE.le m n
    ⊢ Exists fun i => Exists fun j => Membership.mem (setOf fun x => Membership.me …
  -/
  exacts [⟨m, hm, refl_mem_uniformity ha⟩, ⟨n, le_refl n, hn m hm n le_rfl⟩]
  /-
    🎉 no goals
  -/


/-- An auxiliary sequence of sets approximating a Cauchy filter. -/
def setSeqAux (n : ℕ) : { s : Set α // s ∈ f ∧ s ×ˢ s ⊆ U n } :=
  -- Porting note: changed `∃ _ : s ∈ f, ..` to `s ∈ f ∧ ..`
  Classical.indefiniteDescription _ <| (cauchy_iff.1 hf).2 (U n) (U_mem n)


/-- Given a Cauchy filter `f` and a sequence `U` of entourages, `set_seq` provides
an antitone sequence of sets `s n ∈ f` such that `s n ×ˢ s n ⊆ U`. -/
def setSeq (n : ℕ) : Set α :=
  ⋂ m ∈ Set.Iic n, (setSeqAux hf U_mem m).val


theorem setSeq_mem (n : ℕ) : setSeq hf U_mem n ∈ f :=
  (biInter_mem (finite_le_nat n)).2 fun m _ => (setSeqAux hf U_mem m).2.1


theorem setSeq_mono ⦃m n : ℕ⦄ (h : m ≤ n) : setSeq hf U_mem n ⊆ setSeq hf U_mem m :=
  biInter_subset_biInter_left <| Iic_subset_Iic.2 h


theorem setSeq_sub_aux (n : ℕ) : setSeq hf U_mem n ⊆ setSeqAux hf U_mem n :=
  biInter_subset_of_mem right_mem_Iic


theorem setSeq_prod_subset {N m n} (hm : N ≤ m) (hn : N ≤ n) :
    setSeq hf U_mem m ×ˢ setSeq hf U_mem n ⊆ U N := fun p hp => by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    f : Filter α
    hf : Cauchy f
    U : Nat → Set (Prod α α)
    U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
    N m n : Nat
    hm : LE.le N m
    hn : LE.le N n
    p : Prod α α
    hp : Membership.mem (SProd.sprod (SequentiallyComplete.setSeq hf U_mem m) (Seq …
    ⊢ Membership.mem (U N) p
  -/
  refine (setSeqAux hf U_mem N).2.2 ⟨?_, ?_⟩ <;> apply setSeq_sub_aux
    /-
      case refine_1.a
      α : Type u
      uniformSpace : UniformSpace α
      f : Filter α
      hf : Cauchy f
      U : Nat → Set (Prod α α)
      U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
      N m n : Nat
      hm : LE.le N m
      hn : LE.le N n
      p : Prod α α
      hp : Membership.mem (SProd.sprod (SequentiallyComplete.setSeq hf U_mem m) (Seq …
      ⊢ Membership.mem (SequentiallyComplete.setSeq hf U_mem N) p.1
    -/
  · exact setSeq_mono hf U_mem hm hp.1
    /-
      🎉 no goals
    -/
    /-
      case refine_2.a
      α : Type u
      uniformSpace : UniformSpace α
      f : Filter α
      hf : Cauchy f
      U : Nat → Set (Prod α α)
      U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
      N m n : Nat
      hm : LE.le N m
      hn : LE.le N n
      p : Prod α α
      hp : Membership.mem (SProd.sprod (SequentiallyComplete.setSeq hf U_mem m) (Seq …
      ⊢ Membership.mem (SequentiallyComplete.setSeq hf U_mem N) p.2
    -/
  · exact setSeq_mono hf U_mem hn hp.2
    /-
      🎉 no goals
    -/


/-- A sequence of points such that `seq n ∈ setSeq n`. Here `setSeq` is an antitone
sequence of sets `setSeq n ∈ f` with diameters controlled by a given sequence
of entourages. -/
def seq (n : ℕ) : α :=
  (hf.1.nonempty_of_mem (setSeq_mem hf U_mem n)).choose


theorem seq_mem (n : ℕ) : seq hf U_mem n ∈ setSeq hf U_mem n :=
  (hf.1.nonempty_of_mem (setSeq_mem hf U_mem n)).choose_spec


theorem seq_pair_mem ⦃N m n : ℕ⦄ (hm : N ≤ m) (hn : N ≤ n) :
    (seq hf U_mem m, seq hf U_mem n) ∈ U N :=
  setSeq_prod_subset hf U_mem hm hn ⟨seq_mem hf U_mem m, seq_mem hf U_mem n⟩


theorem seq_is_cauchySeq (U_le : ∀ s ∈ 𝓤 α, ∃ n, U n ⊆ s) : CauchySeq <| seq hf U_mem :=
  cauchySeq_of_controlled U U_le <| seq_pair_mem hf U_mem


/-- If the sequence `SequentiallyComplete.seq` converges to `a`, then `f ≤ 𝓝 a`. -/
theorem le_nhds_of_seq_tendsto_nhds (U_le : ∀ s ∈ 𝓤 α, ∃ n, U n ⊆ s)
    ⦃a : α⦄ (ha : Tendsto (seq hf U_mem) atTop (𝓝 a)) : f ≤ 𝓝 a :=
  le_nhds_of_cauchy_adhp_aux
    (fun s hs => by
      /-
        α : Type u
        uniformSpace : UniformSpace α
        f : Filter α
        hf : Cauchy f
        U : Nat → Set (Prod α α)
        U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
        U_le : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n  …
        a : α
        ha : Filter.Tendsto (SequentiallyComplete.seq hf U_mem) Filter.atTop (nhds a)
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        ⊢ Exists fun t => And (Membership.mem f t) (And (HasSubset.Subset (SProd.sprod …
      -/
      rcases U_le s hs with ⟨m, hm⟩
      /-
        case intro
        α : Type u
        uniformSpace : UniformSpace α
        f : Filter α
        hf : Cauchy f
        U : Nat → Set (Prod α α)
        U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
        U_le : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n  …
        a : α
        ha : Filter.Tendsto (SequentiallyComplete.seq hf U_mem) Filter.atTop (nhds a)
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        m : Nat
        hm : HasSubset.Subset (U m) s
        ⊢ Exists fun t => And (Membership.mem f t) (And (HasSubset.Subset (SProd.sprod …
      -/
      rcases tendsto_atTop'.1 ha _ (mem_nhds_left a (U_mem m)) with ⟨n, hn⟩
      refine
        ⟨setSeq hf U_mem (max m n), setSeq_mem hf U_mem _, ?_, seq hf U_mem (max m n), ?_,
          seq_mem hf U_mem _⟩
        /-
          case intro.intro.refine_1
          α : Type u
          uniformSpace : UniformSpace α
          f : Filter α
          hf : Cauchy f
          U : Nat → Set (Prod α α)
          U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
          U_le : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n  …
          a : α
          ha : Filter.Tendsto (SequentiallyComplete.seq hf U_mem) Filter.atTop (nhds a)
          s : Set (Prod α α)
          hs : Membership.mem (uniformity α) s
          m : Nat
          hm : HasSubset.Subset (U m) s
          n : Nat
          hn : ∀ (b : Nat), GE.ge b n → Membership.mem (setOf fun y => Membership.mem (U …
          ⊢ HasSubset.Subset (SProd.sprod (SequentiallyComplete.setSeq hf U_mem (Max.max …
        -/
      · have := le_max_left m n
        /-
          case intro.intro.refine_1
          α : Type u
          uniformSpace : UniformSpace α
          f : Filter α
          hf : Cauchy f
          U : Nat → Set (Prod α α)
          U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
          U_le : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n  …
          a : α
          ha : Filter.Tendsto (SequentiallyComplete.seq hf U_mem) Filter.atTop (nhds a)
          s : Set (Prod α α)
          hs : Membership.mem (uniformity α) s
          m : Nat
          hm : HasSubset.Subset (U m) s
          n : Nat
          hn : ∀ (b : Nat), GE.ge b n → Membership.mem (setOf fun y => Membership.mem (U …
          this : LE.le m (Max.max m n)
          ⊢ HasSubset.Subset (SProd.sprod (SequentiallyComplete.setSeq hf U_mem (Max.max …
        -/
        exact Set.Subset.trans (setSeq_prod_subset hf U_mem this this) hm
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.refine_2
          α : Type u
          uniformSpace : UniformSpace α
          f : Filter α
          hf : Cauchy f
          U : Nat → Set (Prod α α)
          U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
          U_le : ∀ (s : Set (Prod α α)), Membership.mem (uniformity α) s → Exists fun n  …
          a : α
          ha : Filter.Tendsto (SequentiallyComplete.seq hf U_mem) Filter.atTop (nhds a)
          s : Set (Prod α α)
          hs : Membership.mem (uniformity α) s
          m : Nat
          hm : HasSubset.Subset (U m) s
          n : Nat
          hn : ∀ (b : Nat), GE.ge b n → Membership.mem (setOf fun y => Membership.mem (U …
          ⊢ Membership.mem s { fst := a, snd := SequentiallyComplete.seq hf U_mem (Max.m …
        -/
      · exact hm (hn _ <| le_max_right m n))
        /-
          🎉 no goals
        -/


/-- A uniform space is complete provided that (a) its uniformity filter has a countable basis;
(b) any sequence satisfying a "controlled" version of the Cauchy condition converges. -/
theorem complete_of_convergent_controlled_sequences (U : ℕ → Set (α × α)) (U_mem : ∀ n, U n ∈ 𝓤 α)
    (HU : ∀ u : ℕ → α, (∀ N m n, N ≤ m → N ≤ n → (u m, u n) ∈ U N) → ∃ a, Tendsto u atTop (𝓝 a)) :
    CompleteSpace α := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    inst✝ : (uniformity α).IsCountablyGenerated
    U : Nat → Set (Prod α α)
    U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
    HU : ∀ (u : Nat → α), (∀ (N m n : Nat), LE.le N m → LE.le N n → Membership.mem …
    ⊢ CompleteSpace α
  -/
  obtain ⟨U', -, hU'⟩ := (𝓤 α).exists_antitone_seq
  /-
    case intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    inst✝ : (uniformity α).IsCountablyGenerated
    U : Nat → Set (Prod α α)
    U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
    HU : ∀ (u : Nat → α), (∀ (N m n : Nat), LE.le N m → LE.le N n → Membership.mem …
    U' : Nat → Set (Prod α α)
    hU' : ∀ {s : Set (Prod α α)}, Iff (Membership.mem (uniformity α) s) (Exists fu …
    ⊢ CompleteSpace α
  -/
  have Hmem : ∀ n, U n ∩ U' n ∈ 𝓤 α := fun n => inter_mem (U_mem n) (hU'.2 ⟨n, Subset.refl _⟩)
  refine ⟨fun hf => (HU (seq hf Hmem) fun N m n hm hn => ?_).imp <|
    le_nhds_of_seq_tendsto_nhds _ _ fun s hs => ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u
      uniformSpace : UniformSpace α
      inst✝ : (uniformity α).IsCountablyGenerated
      U : Nat → Set (Prod α α)
      U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
      HU : ∀ (u : Nat → α), (∀ (N m n : Nat), LE.le N m → LE.le N n → Membership.mem …
      U' : Nat → Set (Prod α α)
      hU' : ∀ {s : Set (Prod α α)}, Iff (Membership.mem (uniformity α) s) (Exists fu …
      Hmem : ∀ (n : Nat), Membership.mem (uniformity α) (Inter.inter (U n) (U' n))
      f✝ : Filter α
      hf : Cauchy f✝
      N m n : Nat
      hm : LE.le N m
      hn : LE.le N n
      ⊢ Membership.mem (U N) { fst := SequentiallyComplete.seq hf Hmem m, snd := Seq …
    -/
  · exact inter_subset_left (seq_pair_mem hf Hmem hm hn)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u
      uniformSpace : UniformSpace α
      inst✝ : (uniformity α).IsCountablyGenerated
      U : Nat → Set (Prod α α)
      U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
      HU : ∀ (u : Nat → α), (∀ (N m n : Nat), LE.le N m → LE.le N n → Membership.mem …
      U' : Nat → Set (Prod α α)
      hU' : ∀ {s : Set (Prod α α)}, Iff (Membership.mem (uniformity α) s) (Exists fu …
      Hmem : ∀ (n : Nat), Membership.mem (uniformity α) (Inter.inter (U n) (U' n))
      f✝ : Filter α
      hf : Cauchy f✝
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      ⊢ Exists fun n => HasSubset.Subset (Inter.inter (U n) (U' n)) s
    -/
  · rcases hU'.1 hs with ⟨N, hN⟩
    /-
      case intro.intro.refine_2.intro
      α : Type u
      uniformSpace : UniformSpace α
      inst✝ : (uniformity α).IsCountablyGenerated
      U : Nat → Set (Prod α α)
      U_mem : ∀ (n : Nat), Membership.mem (uniformity α) (U n)
      HU : ∀ (u : Nat → α), (∀ (N m n : Nat), LE.le N m → LE.le N n → Membership.mem …
      U' : Nat → Set (Prod α α)
      hU' : ∀ {s : Set (Prod α α)}, Iff (Membership.mem (uniformity α) s) (Exists fu …
      Hmem : ∀ (n : Nat), Membership.mem (uniformity α) (Inter.inter (U n) (U' n))
      f✝ : Filter α
      hf : Cauchy f✝
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      N : Nat
      hN : HasSubset.Subset (U' N) s
      ⊢ Exists fun n => HasSubset.Subset (Inter.inter (U n) (U' n)) s
    -/
    exact ⟨N, Subset.trans inter_subset_right hN⟩
    /-
      🎉 no goals
    -/


/-- A sequentially complete uniform space with a countable basis of the uniformity filter is
complete. -/
theorem complete_of_cauchySeq_tendsto (H' : ∀ u : ℕ → α, CauchySeq u → ∃ a, Tendsto u atTop (𝓝 a)) :
    CompleteSpace α :=
  let ⟨U', _, hU'⟩ := (𝓤 α).exists_antitone_seq
  complete_of_convergent_controlled_sequences U' (fun n => hU'.2 ⟨n, Subset.refl _⟩) fun u hu =>
    H' u <| cauchySeq_of_controlled U' (fun _ hs => hU'.1 hs) hu


instance (priority := 100) firstCountableTopology : FirstCountableTopology α :=
               /-
                 α : Type u
                 β : Type v
                 uniformSpace : UniformSpace α
                 inst✝ : (uniformity α).IsCountablyGenerated
                 a : α
                 ⊢ (nhds a).IsCountablyGenerated
               -/
  ⟨fun a => by rw [nhds_eq_comap_uniformity]; infer_instance⟩
                                              /-
                                                🎉 no goals
                                              -/


/-- A separable uniform space with countably generated uniformity filter is second countable:
one obtains a countable basis by taking the balls centered at points in a dense subset,
and with rational "radii" from a countable open symmetric antitone basis of `𝓤 α`. We do not
register this as an instance, as there is already an instance going in the other direction
from second countable spaces to separable spaces, and we want to avoid loops. -/
theorem secondCountable_of_separable [SeparableSpace α] : SecondCountableTopology α := by
  /-
    α : Type u
    uniformSpace : UniformSpace α
    inst✝¹ : (uniformity α).IsCountablyGenerated
    inst✝ : TopologicalSpace.SeparableSpace α
    ⊢ SecondCountableTopology α
  -/
  rcases exists_countable_dense α with ⟨s, hsc, hsd⟩
  obtain
    ⟨t : ℕ → Set (α × α), hto : ∀ i : ℕ, t i ∈ (𝓤 α).sets ∧ IsOpen (t i) ∧ SymmetricRel (t i),
      h_basis : (𝓤 α).HasAntitoneBasis t⟩ :=
    (@uniformity_hasBasis_open_symmetric α _).exists_antitone_subbasis
  /-
    case intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    inst✝¹ : (uniformity α).IsCountablyGenerated
    inst✝ : TopologicalSpace.SeparableSpace α
    s : Set α
    hsc : s.Countable
    hsd : Dense s
    t : Nat → Set (Prod α α)
    hto : ∀ (i : Nat), And (Membership.mem (uniformity α).sets (t i)) (And (IsOpen …
    h_basis : (uniformity α).HasAntitoneBasis t
    ⊢ SecondCountableTopology α
  -/
  choose ht_mem hto hts using hto
  /-
    case intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    inst✝¹ : (uniformity α).IsCountablyGenerated
    inst✝ : TopologicalSpace.SeparableSpace α
    s : Set α
    hsc : s.Countable
    hsd : Dense s
    t : Nat → Set (Prod α α)
    h_basis : (uniformity α).HasAntitoneBasis t
    ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
    hto : ∀ (i : Nat), IsOpen (t i)
    hts : ∀ (i : Nat), SymmetricRel (t i)
    ⊢ SecondCountableTopology α
  -/
  refine ⟨⟨⋃ x ∈ s, range fun k => ball x (t k), hsc.biUnion fun x _ => countable_range _, ?_⟩⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    uniformSpace : UniformSpace α
    inst✝¹ : (uniformity α).IsCountablyGenerated
    inst✝ : TopologicalSpace.SeparableSpace α
    s : Set α
    hsc : s.Countable
    hsd : Dense s
    t : Nat → Set (Prod α α)
    h_basis : (uniformity α).HasAntitoneBasis t
    ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
    hto : ∀ (i : Nat), IsOpen (t i)
    hts : ∀ (i : Nat), SymmetricRel (t i)
    ⊢ Eq UniformSpace.toTopologicalSpace (TopologicalSpace.generateFrom (Set.iUnio …
  -/
  refine (isTopologicalBasis_of_isOpen_of_nhds ?_ ?_).eq_generateFrom
    /-
      case intro.intro.intro.intro.refine_1
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      ⊢ ∀ (u : Set α), Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Set.r …
    -/
  · simp only [mem_iUnion₂, mem_range]
    /-
      case intro.intro.intro.intro.refine_1
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      ⊢ ∀ (u : Set α), (Exists fun i => Exists fun h => Exists fun y => Eq (UniformS …
    -/
    rintro _ ⟨x, _, k, rfl⟩
    /-
      case intro.intro.intro.intro.refine_1.intro.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      w✝ : Membership.mem s x
      k : Nat
      ⊢ IsOpen (UniformSpace.ball x (t k))
    -/
    exact isOpen_ball x (hto k)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      ⊢ ∀ (a : α) (u : Set α), Membership.mem u a → IsOpen u → Exists fun v => And ( …
    -/
  · intro x V hxV hVo
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      V : Set α
      hxV : Membership.mem V x
      hVo : IsOpen V
      ⊢ Exists fun v => And (Membership.mem (Set.iUnion fun x => Set.iUnion fun h => …
    -/
    simp only [mem_iUnion₂, mem_range, exists_prop]
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      V : Set α
      hxV : Membership.mem V x
      hVo : IsOpen V
      ⊢ Exists fun v => And (Exists fun i => And (Membership.mem s i) (Exists fun y  …
    -/
    rcases UniformSpace.mem_nhds_iff.1 (IsOpen.mem_nhds hVo hxV) with ⟨U, hU, hUV⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      V : Set α
      hxV : Membership.mem V x
      hVo : IsOpen V
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUV : HasSubset.Subset (UniformSpace.ball x U) V
      ⊢ Exists fun v => And (Exists fun i => And (Membership.mem s i) (Exists fun y  …
    -/
    rcases comp_symm_of_uniformity hU with ⟨U', hU', _, hUU'⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro.intro
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      V : Set α
      hxV : Membership.mem V x
      hVo : IsOpen V
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUV : HasSubset.Subset (UniformSpace.ball x U) V
      U' : Set (Prod α α)
      hU' : Membership.mem (uniformity α) U'
      left✝ : ∀ {a b : α}, Membership.mem U' { fst := a, snd := b } → Membership.mem …
      hUU' : HasSubset.Subset (compRel U' U') U
      ⊢ Exists fun v => And (Exists fun i => And (Membership.mem s i) (Exists fun y  …
    -/
    rcases h_basis.toHasBasis.mem_iff.1 hU' with ⟨k, -, hk⟩
    rcases hsd.inter_open_nonempty (ball x <| t k) (isOpen_ball x (hto k))
        ⟨x, UniformSpace.mem_ball_self _ (ht_mem k)⟩ with
      ⟨y, hxy, hys⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro.intro.intro.intr …
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      V : Set α
      hxV : Membership.mem V x
      hVo : IsOpen V
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUV : HasSubset.Subset (UniformSpace.ball x U) V
      U' : Set (Prod α α)
      hU' : Membership.mem (uniformity α) U'
      left✝ : ∀ {a b : α}, Membership.mem U' { fst := a, snd := b } → Membership.mem …
      hUU' : HasSubset.Subset (compRel U' U') U
      k : Nat
      hk : HasSubset.Subset (t k) U'
      y : α
      hxy : Membership.mem (UniformSpace.ball x (t k)) y
      hys : Membership.mem s y
      ⊢ Exists fun v => And (Exists fun i => And (Membership.mem s i) (Exists fun y  …
    -/
    refine ⟨_, ⟨y, hys, k, rfl⟩, (hts k).subset hxy, fun z hz => ?_⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro.intro.intro.intr …
      α : Type u
      uniformSpace : UniformSpace α
      inst✝¹ : (uniformity α).IsCountablyGenerated
      inst✝ : TopologicalSpace.SeparableSpace α
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      t : Nat → Set (Prod α α)
      h_basis : (uniformity α).HasAntitoneBasis t
      ht_mem : ∀ (i : Nat), Membership.mem (uniformity α).sets (t i)
      hto : ∀ (i : Nat), IsOpen (t i)
      hts : ∀ (i : Nat), SymmetricRel (t i)
      x : α
      V : Set α
      hxV : Membership.mem V x
      hVo : IsOpen V
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      hUV : HasSubset.Subset (UniformSpace.ball x U) V
      U' : Set (Prod α α)
      hU' : Membership.mem (uniformity α) U'
      left✝ : ∀ {a b : α}, Membership.mem U' { fst := a, snd := b } → Membership.mem …
      hUU' : HasSubset.Subset (compRel U' U') U
      k : Nat
      hk : HasSubset.Subset (t k) U'
      y : α
      hxy : Membership.mem (UniformSpace.ball x (t k)) y
      hys : Membership.mem s y
      z : α
      hz : Membership.mem (UniformSpace.ball y (t k)) z
      ⊢ Membership.mem V z
    -/
    exact hUV (ball_subset_of_comp_subset (hk hxy) hUU' (hk hz))
    /-
      🎉 no goals
    -/


/-- A Cauchy filter in a discrete uniform space is contained in a principal filter-/
theorem DiscreteUnif.cauchy_le_pure {X : Type*} {uX : UniformSpace X}
    (hX : uX = ⊥) {α : Filter X} (hα : Cauchy α) : ∃ x : X, α = pure x := by
  /-
    X : Type u_1
    uX : UniformSpace X
    hX : Eq uX Bot.bot
    α : Filter X
    hα : Cauchy α
    ⊢ Exists fun x => Eq α (Pure.pure x)
  -/
  rcases hα with ⟨α_ne_bot, α_le⟩
  /-
    case intro
    X : Type u_1
    uX : UniformSpace X
    hX : Eq uX Bot.bot
    α : Filter X
    α_ne_bot : α.NeBot
    α_le : LE.le (SProd.sprod α α) (uniformity X)
    ⊢ Exists fun x => Eq α (Pure.pure x)
  -/
  rw [hX, bot_uniformity, le_principal_iff, mem_prod_iff] at α_le
  /-
    case intro
    X : Type u_1
    uX : UniformSpace X
    hX : Eq uX Bot.bot
    α : Filter X
    α_ne_bot : α.NeBot
    α_le : Exists fun t₁ => And (Membership.mem α t₁) (Exists fun t₂ => And (Membe …
    ⊢ Exists fun x => Eq α (Pure.pure x)
  -/
  obtain ⟨S, ⟨hS, ⟨T, ⟨hT, H⟩⟩⟩⟩ := α_le
  obtain ⟨x, rfl⟩ := eq_singleton_left_of_prod_subset_idRel (α_ne_bot.nonempty_of_mem hS)
    (Filter.nonempty_of_mem hT) H
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    uX : UniformSpace X
    hX : Eq uX Bot.bot
    α : Filter X
    α_ne_bot : α.NeBot
    T : Set X
    hT : Membership.mem α T
    x : X
    hS : Membership.mem α (Singleton.singleton x)
    H : HasSubset.Subset (SProd.sprod (Singleton.singleton x) T) idRel
    ⊢ Exists fun x => Eq α (Pure.pure x)
  -/
  exact ⟨x, α_ne_bot.le_pure_iff.mp <| le_pure_iff.mpr hS⟩
  /-
    🎉 no goals
  -/


/-- A constant to which a Cauchy filter in a discrete uniform space converges. -/
noncomputable def DiscreteUnif.cauchyConst {X : Type*} {uX : UniformSpace X}
    (hX : uX = ⊥) {α : Filter X} (hα : Cauchy α) : X :=
  (DiscreteUnif.cauchy_le_pure hX hα).choose


theorem DiscreteUnif.eq_const_of_cauchy {X : Type*} {uX : UniformSpace X} (hX : uX = ⊥)
    {α : Filter X} (hα : Cauchy α) : α = pure (DiscreteUnif.cauchyConst hX hα) :=
  (DiscreteUnif.cauchy_le_pure hX hα).choose_spec


