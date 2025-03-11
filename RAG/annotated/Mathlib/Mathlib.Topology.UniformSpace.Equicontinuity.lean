/-- A family `F : ι → X → α` of functions from a topological space to a uniform space is
*equicontinuous at `x₀ : X`* if, for all entourages `U ∈ 𝓤 α`, there is a neighborhood `V` of `x₀`
such that, for all `x ∈ V` and for all `i : ι`, `F i x` is `U`-close to `F i x₀`. -/
def EquicontinuousAt (F : ι → X → α) (x₀ : X) : Prop :=
  ∀ U ∈ 𝓤 α, ∀ᶠ x in 𝓝 x₀, ∀ i, (F i x₀, F i x) ∈ U


/-- We say that a set `H : Set (X → α)` of functions is equicontinuous at a point if the family
`(↑) : ↥H → (X → α)` is equicontinuous at that point. -/
protected abbrev Set.EquicontinuousAt (H : Set <| X → α) (x₀ : X) : Prop :=
  EquicontinuousAt ((↑) : H → X → α) x₀


/-- A family `F : ι → X → α` of functions from a topological space to a uniform space is
*equicontinuous at `x₀ : X` within `S : Set X`* if, for all entourages `U ∈ 𝓤 α`, there is a
neighborhood `V` of `x₀` within `S` such that, for all `x ∈ V` and for all `i : ι`, `F i x` is
`U`-close to `F i x₀`. -/
def EquicontinuousWithinAt (F : ι → X → α) (S : Set X) (x₀ : X) : Prop :=
  ∀ U ∈ 𝓤 α, ∀ᶠ x in 𝓝[S] x₀, ∀ i, (F i x₀, F i x) ∈ U


/-- We say that a set `H : Set (X → α)` of functions is equicontinuous at a point within a subset
if the family `(↑) : ↥H → (X → α)` is equicontinuous at that point within that same subset. -/
protected abbrev Set.EquicontinuousWithinAt (H : Set <| X → α) (S : Set X) (x₀ : X) : Prop :=
  EquicontinuousWithinAt ((↑) : H → X → α) S x₀


/-- A family `F : ι → X → α` of functions from a topological space to a uniform space is
*equicontinuous* on all of `X` if it is equicontinuous at each point of `X`. -/
def Equicontinuous (F : ι → X → α) : Prop :=
  ∀ x₀, EquicontinuousAt F x₀


/-- We say that a set `H : Set (X → α)` of functions is equicontinuous if the family
`(↑) : ↥H → (X → α)` is equicontinuous. -/
protected abbrev Set.Equicontinuous (H : Set <| X → α) : Prop :=
  Equicontinuous ((↑) : H → X → α)


/-- A family `F : ι → X → α` of functions from a topological space to a uniform space is
*equicontinuous on `S : Set X`* if it is equicontinuous *within `S`* at each point of `S`. -/
def EquicontinuousOn (F : ι → X → α) (S : Set X) : Prop :=
  ∀ x₀ ∈ S, EquicontinuousWithinAt F S x₀


/-- We say that a set `H : Set (X → α)` of functions is equicontinuous on a subset if the family
`(↑) : ↥H → (X → α)` is equicontinuous on that subset. -/
protected abbrev Set.EquicontinuousOn (H : Set <| X → α) (S : Set X) : Prop :=
  EquicontinuousOn ((↑) : H → X → α) S


/-- A family `F : ι → β → α` of functions between uniform spaces is *uniformly equicontinuous* if,
for all entourages `U ∈ 𝓤 α`, there is an entourage `V ∈ 𝓤 β` such that, whenever `x` and `y` are
`V`-close, we have that, *for all `i : ι`*, `F i x` is `U`-close to `F i y`. -/
def UniformEquicontinuous (F : ι → β → α) : Prop :=
  ∀ U ∈ 𝓤 α, ∀ᶠ xy : β × β in 𝓤 β, ∀ i, (F i xy.1, F i xy.2) ∈ U


/-- We say that a set `H : Set (X → α)` of functions is uniformly equicontinuous if the family
`(↑) : ↥H → (X → α)` is uniformly equicontinuous. -/
protected abbrev Set.UniformEquicontinuous (H : Set <| β → α) : Prop :=
  UniformEquicontinuous ((↑) : H → β → α)


/-- A family `F : ι → β → α` of functions between uniform spaces is
*uniformly equicontinuous on `S : Set β`* if, for all entourages `U ∈ 𝓤 α`, there is a relative
entourage `V ∈ 𝓤 β ⊓ 𝓟 (S ×ˢ S)` such that, whenever `x` and `y` are `V`-close, we have that,
*for all `i : ι`*, `F i x` is `U`-close to `F i y`. -/
def UniformEquicontinuousOn (F : ι → β → α) (S : Set β) : Prop :=
  ∀ U ∈ 𝓤 α, ∀ᶠ xy : β × β in 𝓤 β ⊓ 𝓟 (S ×ˢ S), ∀ i, (F i xy.1, F i xy.2) ∈ U


/-- We say that a set `H : Set (X → α)` of functions is uniformly equicontinuous on a subset if the
family `(↑) : ↥H → (X → α)` is uniformly equicontinuous on that subset. -/
protected abbrev Set.UniformEquicontinuousOn (H : Set <| β → α) (S : Set β) : Prop :=
  UniformEquicontinuousOn ((↑) : H → β → α) S


lemma EquicontinuousAt.equicontinuousWithinAt {F : ι → X → α} {x₀ : X} (H : EquicontinuousAt F x₀)
    (S : Set X) : EquicontinuousWithinAt F S x₀ :=
  fun U hU ↦ (H U hU).filter_mono inf_le_left


lemma EquicontinuousWithinAt.mono {F : ι → X → α} {x₀ : X} {S T : Set X}
    (H : EquicontinuousWithinAt F T x₀) (hST : S ⊆ T) : EquicontinuousWithinAt F S x₀ :=
  fun U hU ↦ (H U hU).filter_mono <| nhdsWithin_mono x₀ hST


@[simp] lemma equicontinuousWithinAt_univ (F : ι → X → α) (x₀ : X) :
    EquicontinuousWithinAt F univ x₀ ↔ EquicontinuousAt F x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    x₀ : X
    ⊢ Iff (EquicontinuousWithinAt F Set.univ x₀) (EquicontinuousAt F x₀)
  -/
  rw [EquicontinuousWithinAt, EquicontinuousAt, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


lemma equicontinuousAt_restrict_iff (F : ι → X → α) {S : Set X} (x₀ : S) :
    EquicontinuousAt (S.restrict ∘ F) x₀ ↔ EquicontinuousWithinAt F S x₀ := by
  simp [EquicontinuousWithinAt, EquicontinuousAt,
    ← eventually_nhds_subtype_iff]


lemma Equicontinuous.equicontinuousOn {F : ι → X → α} (H : Equicontinuous F)
    (S : Set X) : EquicontinuousOn F S :=
  fun x _ ↦ (H x).equicontinuousWithinAt S


lemma EquicontinuousOn.mono {F : ι → X → α} {S T : Set X}
    (H : EquicontinuousOn F T) (hST : S ⊆ T) : EquicontinuousOn F S :=
  fun x hx ↦ (H x (hST hx)).mono hST


lemma equicontinuousOn_univ (F : ι → X → α) :
    EquicontinuousOn F univ ↔ Equicontinuous F := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    ⊢ Iff (EquicontinuousOn F Set.univ) (Equicontinuous F)
  -/
  simp [EquicontinuousOn, Equicontinuous]
  /-
    🎉 no goals
  -/


lemma equicontinuous_restrict_iff (F : ι → X → α) {S : Set X} :
    Equicontinuous (S.restrict ∘ F) ↔ EquicontinuousOn F S := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    S : Set X
    ⊢ Iff (Equicontinuous (Function.comp S.restrict F)) (EquicontinuousOn F S)
  -/
  simp [Equicontinuous, EquicontinuousOn, equicontinuousAt_restrict_iff]
  /-
    🎉 no goals
  -/


lemma UniformEquicontinuous.uniformEquicontinuousOn {F : ι → β → α} (H : UniformEquicontinuous F)
    (S : Set β) : UniformEquicontinuousOn F S :=
  fun U hU ↦ (H U hU).filter_mono inf_le_left


lemma UniformEquicontinuousOn.mono {F : ι → β → α} {S T : Set β}
    (H : UniformEquicontinuousOn F T) (hST : S ⊆ T) : UniformEquicontinuousOn F S :=
                                        /-
                                          ι : Type u_1
                                          α : Type u_6
                                          β : Type u_8
                                          uα : UniformSpace α
                                          uβ : UniformSpace β
                                          F : ι → β → α
                                          S T : Set β
                                          H : UniformEquicontinuousOn F T
                                          hST : HasSubset.Subset S T
                                          U : Set (Prod α α)
                                          hU : Membership.mem (uniformity α) U
                                          ⊢ LE.le (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))) (Min.min …
                                        -/
  fun U hU ↦ (H U hU).filter_mono <| by gcongr
                                        /-
                                          🎉 no goals
                                        -/


lemma uniformEquicontinuousOn_univ (F : ι → β → α) :
    UniformEquicontinuousOn F univ ↔ UniformEquicontinuous F := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    ⊢ Iff (UniformEquicontinuousOn F Set.univ) (UniformEquicontinuous F)
  -/
  simp [UniformEquicontinuousOn, UniformEquicontinuous]
  /-
    🎉 no goals
  -/


lemma uniformEquicontinuous_restrict_iff (F : ι → β → α) {S : Set β} :
    UniformEquicontinuous (S.restrict ∘ F) ↔ UniformEquicontinuousOn F S := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    S : Set β
    ⊢ Iff (UniformEquicontinuous (Function.comp S.restrict F)) (UniformEquicontinu …
  -/
  rw [UniformEquicontinuous, UniformEquicontinuousOn]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    S : Set β
    ⊢ Iff (∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Filter.Eventu …
  -/
  conv in _ ⊓ _ => rw [← Subtype.range_val (s := S), ← range_prod_map, ← map_comap]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    S : Set β
    ⊢ Iff (∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Filter.Eventu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma equicontinuousAt_empty [h : IsEmpty ι] (F : ι → X → α) (x₀ : X) :
    EquicontinuousAt F x₀ :=
  fun _ _ ↦ Eventually.of_forall (fun _ ↦ h.elim)


@[simp]
lemma equicontinuousWithinAt_empty [h : IsEmpty ι] (F : ι → X → α) (S : Set X) (x₀ : X) :
    EquicontinuousWithinAt F S x₀ :=
  fun _ _ ↦ Eventually.of_forall (fun _ ↦ h.elim)


@[simp]
lemma equicontinuous_empty [IsEmpty ι] (F : ι → X → α) :
    Equicontinuous F :=
  equicontinuousAt_empty F


@[simp]
lemma equicontinuousOn_empty [IsEmpty ι] (F : ι → X → α) (S : Set X) :
    EquicontinuousOn F S :=
  fun x₀ _ ↦ equicontinuousWithinAt_empty F S x₀


@[simp]
lemma uniformEquicontinuous_empty [h : IsEmpty ι] (F : ι → β → α) :
    UniformEquicontinuous F :=
  fun _ _ ↦ Eventually.of_forall (fun _ ↦ h.elim)


@[simp]
lemma uniformEquicontinuousOn_empty [h : IsEmpty ι] (F : ι → β → α) (S : Set β) :
    UniformEquicontinuousOn F S :=
  fun _ _ ↦ Eventually.of_forall (fun _ ↦ h.elim)


theorem equicontinuousAt_finite [Finite ι] {F : ι → X → α} {x₀ : X} :
    EquicontinuousAt F x₀ ↔ ∀ i, ContinuousAt (F i) x₀ := by
  simp [EquicontinuousAt, ContinuousAt, (nhds_basis_uniformity' (𝓤 α).basis_sets).tendsto_right_iff,
    UniformSpace.ball, @forall_swap _ ι]


theorem equicontinuousWithinAt_finite [Finite ι] {F : ι → X → α} {S : Set X} {x₀ : X} :
    EquicontinuousWithinAt F S x₀ ↔ ∀ i, ContinuousWithinAt (F i) S x₀ := by
  simp [EquicontinuousWithinAt, ContinuousWithinAt,
    (nhds_basis_uniformity' (𝓤 α).basis_sets).tendsto_right_iff, UniformSpace.ball,
    @forall_swap _ ι]


theorem equicontinuous_finite [Finite ι] {F : ι → X → α} :
    Equicontinuous F ↔ ∀ i, Continuous (F i) := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    inst✝ : Finite ι
    F : ι → X → α
    ⊢ Iff (Equicontinuous F) (∀ (i : ι), Continuous (F i))
  -/
  simp only [Equicontinuous, equicontinuousAt_finite, continuous_iff_continuousAt, @forall_swap ι]
  /-
    🎉 no goals
  -/


theorem equicontinuousOn_finite [Finite ι] {F : ι → X → α} {S : Set X} :
    EquicontinuousOn F S ↔ ∀ i, ContinuousOn (F i) S := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    inst✝ : Finite ι
    F : ι → X → α
    S : Set X
    ⊢ Iff (EquicontinuousOn F S) (∀ (i : ι), ContinuousOn (F i) S)
  -/
  simp only [EquicontinuousOn, equicontinuousWithinAt_finite, ContinuousOn, @forall_swap ι]
  /-
    🎉 no goals
  -/


theorem uniformEquicontinuous_finite [Finite ι] {F : ι → β → α} :
    UniformEquicontinuous F ↔ ∀ i, UniformContinuous (F i) := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    inst✝ : Finite ι
    F : ι → β → α
    ⊢ Iff (UniformEquicontinuous F) (∀ (i : ι), UniformContinuous (F i))
  -/
  simp only [UniformEquicontinuous, eventually_all, @forall_swap _ ι]; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem uniformEquicontinuousOn_finite [Finite ι] {F : ι → β → α} {S : Set β} :
    UniformEquicontinuousOn F S ↔ ∀ i, UniformContinuousOn (F i) S := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    inst✝ : Finite ι
    F : ι → β → α
    S : Set β
    ⊢ Iff (UniformEquicontinuousOn F S) (∀ (i : ι), UniformContinuousOn (F i) S)
  -/
  simp only [UniformEquicontinuousOn, eventually_all, @forall_swap _ ι]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem equicontinuousAt_unique [Unique ι] {F : ι → X → α} {x : X} :
    EquicontinuousAt F x ↔ ContinuousAt (F default) x :=
  equicontinuousAt_finite.trans Unique.forall_iff


theorem equicontinuousWithinAt_unique [Unique ι] {F : ι → X → α} {S : Set X} {x : X} :
    EquicontinuousWithinAt F S x ↔ ContinuousWithinAt (F default) S x :=
  equicontinuousWithinAt_finite.trans Unique.forall_iff


theorem equicontinuous_unique [Unique ι] {F : ι → X → α} :
    Equicontinuous F ↔ Continuous (F default) :=
  equicontinuous_finite.trans Unique.forall_iff


theorem equicontinuousOn_unique [Unique ι] {F : ι → X → α} {S : Set X} :
    EquicontinuousOn F S ↔ ContinuousOn (F default) S :=
  equicontinuousOn_finite.trans Unique.forall_iff


theorem uniformEquicontinuous_unique [Unique ι] {F : ι → β → α} :
    UniformEquicontinuous F ↔ UniformContinuous (F default) :=
  uniformEquicontinuous_finite.trans Unique.forall_iff


theorem uniformEquicontinuousOn_unique [Unique ι] {F : ι → β → α} {S : Set β} :
    UniformEquicontinuousOn F S ↔ UniformContinuousOn (F default) S :=
  uniformEquicontinuousOn_finite.trans Unique.forall_iff


/-- Reformulation of equicontinuity at `x₀` within a set `S`, comparing two variables near `x₀`
instead of comparing only one with `x₀`. -/
theorem equicontinuousWithinAt_iff_pair {F : ι → X → α} {S : Set X} {x₀ : X} (hx₀ : x₀ ∈ S) :
    EquicontinuousWithinAt F S x₀ ↔
      ∀ U ∈ 𝓤 α, ∃ V ∈ 𝓝[S] x₀, ∀ x ∈ V, ∀ y ∈ V, ∀ i, (F i x, F i y) ∈ U := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    S : Set X
    x₀ : X
    hx₀ : Membership.mem S x₀
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (∀ (U : Set (Prod α α)), Membership.mem  …
  -/
  constructor <;> intro H U hU
    /-
      case mp
      ι : Type u_1
      X : Type u_3
      α : Type u_6
      tX : TopologicalSpace X
      uα : UniformSpace α
      F : ι → X → α
      S : Set X
      x₀ : X
      hx₀ : Membership.mem S x₀
      H : EquicontinuousWithinAt F S x₀
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      ⊢ Exists fun V => And (Membership.mem (nhdsWithin x₀ S) V) (∀ (x : X), Members …
    -/
  · rcases comp_symm_mem_uniformity_sets hU with ⟨V, hV, hVsymm, hVU⟩
    /-
      case mp.intro.intro.intro
      ι : Type u_1
      X : Type u_3
      α : Type u_6
      tX : TopologicalSpace X
      uα : UniformSpace α
      F : ι → X → α
      S : Set X
      x₀ : X
      hx₀ : Membership.mem S x₀
      H : EquicontinuousWithinAt F S x₀
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      V : Set (Prod α α)
      hV : Membership.mem (uniformity α) V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      ⊢ Exists fun V => And (Membership.mem (nhdsWithin x₀ S) V) (∀ (x : X), Members …
    -/
    refine ⟨_, H V hV, fun x hx y hy i => hVU (prod_mk_mem_compRel ?_ (hy i))⟩
    /-
      case mp.intro.intro.intro
      ι : Type u_1
      X : Type u_3
      α : Type u_6
      tX : TopologicalSpace X
      uα : UniformSpace α
      F : ι → X → α
      S : Set X
      x₀ : X
      hx₀ : Membership.mem S x₀
      H : EquicontinuousWithinAt F S x₀
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      V : Set (Prod α α)
      hV : Membership.mem (uniformity α) V
      hVsymm : SymmetricRel V
      hVU : HasSubset.Subset (compRel V V) U
      x : X
      hx : Membership.mem (setOf fun x => (fun x => ∀ (i : ι), Membership.mem V { fs …
      y : X
      hy : Membership.mem (setOf fun x => (fun x => ∀ (i : ι), Membership.mem V { fs …
      i : ι
      ⊢ Membership.mem V { fst := F i x, snd := F i x₀ }
    -/
    exact hVsymm.mk_mem_comm.mp (hx i)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      X : Type u_3
      α : Type u_6
      tX : TopologicalSpace X
      uα : UniformSpace α
      F : ι → X → α
      S : Set X
      x₀ : X
      hx₀ : Membership.mem S x₀
      H : ∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Exists fun V =>  …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      ⊢ Filter.Eventually (fun x => ∀ (i : ι), Membership.mem U { fst := F i x₀, snd …
    -/
  · rcases H U hU with ⟨V, hV, hVU⟩
    /-
      case mpr.intro.intro
      ι : Type u_1
      X : Type u_3
      α : Type u_6
      tX : TopologicalSpace X
      uα : UniformSpace α
      F : ι → X → α
      S : Set X
      x₀ : X
      hx₀ : Membership.mem S x₀
      H : ∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Exists fun V =>  …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      V : Set X
      hV : Membership.mem (nhdsWithin x₀ S) V
      hVU : ∀ (x : X), Membership.mem V x → ∀ (y : X), Membership.mem V y → ∀ (i : ι …
      ⊢ Filter.Eventually (fun x => ∀ (i : ι), Membership.mem U { fst := F i x₀, snd …
    -/
    filter_upwards [hV] using fun x hx i => hVU x₀ (mem_of_mem_nhdsWithin hx₀ hV) x hx i
    /-
      🎉 no goals
    -/


/-- Reformulation of equicontinuity at `x₀` comparing two variables near `x₀` instead of comparing
only one with `x₀`. -/
theorem equicontinuousAt_iff_pair {F : ι → X → α} {x₀ : X} :
    EquicontinuousAt F x₀ ↔
      ∀ U ∈ 𝓤 α, ∃ V ∈ 𝓝 x₀, ∀ x ∈ V, ∀ y ∈ V, ∀ i, (F i x, F i y) ∈ U := by
  simp_rw [← equicontinuousWithinAt_univ, equicontinuousWithinAt_iff_pair (mem_univ x₀),
    nhdsWithin_univ]


/-- Uniform equicontinuity implies equicontinuity. -/
theorem UniformEquicontinuous.equicontinuous {F : ι → β → α} (h : UniformEquicontinuous F) :
    Equicontinuous F := fun x₀ U hU ↦
  mem_of_superset (ball_mem_nhds x₀ (h U hU)) fun _ hx i ↦ hx i


/-- Uniform equicontinuity on a subset implies equicontinuity on that subset. -/
theorem UniformEquicontinuousOn.equicontinuousOn {F : ι → β → α} {S : Set β}
    (h : UniformEquicontinuousOn F S) :
    EquicontinuousOn F S := fun _ hx₀ U hU ↦
  mem_of_superset (ball_mem_nhdsWithin hx₀ (h U hU)) fun _ hx i ↦ hx i


/-- Each function of a family equicontinuous at `x₀` is continuous at `x₀`. -/
theorem EquicontinuousAt.continuousAt {F : ι → X → α} {x₀ : X} (h : EquicontinuousAt F x₀) (i : ι) :
    ContinuousAt (F i) x₀ :=
  (UniformSpace.hasBasis_nhds _).tendsto_right_iff.2 fun U ⟨hU, _⟩ ↦ (h U hU).mono fun _x hx ↦ hx i


/-- Each function of a family equicontinuous at `x₀` within `S` is continuous at `x₀` within `S`. -/
theorem EquicontinuousWithinAt.continuousWithinAt {F : ι → X → α} {S : Set X} {x₀ : X}
    (h : EquicontinuousWithinAt F S x₀) (i : ι) :
    ContinuousWithinAt (F i) S x₀ :=
  (UniformSpace.hasBasis_nhds _).tendsto_right_iff.2 fun U ⟨hU, _⟩ ↦ (h U hU).mono fun _x hx ↦ hx i


protected theorem Set.EquicontinuousAt.continuousAt_of_mem {H : Set <| X → α} {x₀ : X}
    (h : H.EquicontinuousAt x₀) {f : X → α} (hf : f ∈ H) : ContinuousAt f x₀ :=
  h.continuousAt ⟨f, hf⟩


protected theorem Set.EquicontinuousWithinAt.continuousWithinAt_of_mem {H : Set <| X → α}
    {S : Set X} {x₀ : X} (h : H.EquicontinuousWithinAt S x₀) {f : X → α} (hf : f ∈ H) :
    ContinuousWithinAt f S x₀ :=
  h.continuousWithinAt ⟨f, hf⟩


/-- Each function of an equicontinuous family is continuous. -/
theorem Equicontinuous.continuous {F : ι → X → α} (h : Equicontinuous F) (i : ι) :
    Continuous (F i) :=
  continuous_iff_continuousAt.mpr fun x => (h x).continuousAt i


/-- Each function of a family equicontinuous on `S` is continuous on `S`. -/
theorem EquicontinuousOn.continuousOn {F : ι → X → α} {S : Set X} (h : EquicontinuousOn F S)
    (i : ι) : ContinuousOn (F i) S :=
  fun x hx ↦ (h x hx).continuousWithinAt i


protected theorem Set.Equicontinuous.continuous_of_mem {H : Set <| X → α} (h : H.Equicontinuous)
    {f : X → α} (hf : f ∈ H) : Continuous f :=
  h.continuous ⟨f, hf⟩


protected theorem Set.EquicontinuousOn.continuousOn_of_mem {H : Set <| X → α} {S : Set X}
    (h : H.EquicontinuousOn S) {f : X → α} (hf : f ∈ H) : ContinuousOn f S :=
  h.continuousOn ⟨f, hf⟩


/-- Each function of a uniformly equicontinuous family is uniformly continuous. -/
theorem UniformEquicontinuous.uniformContinuous {F : ι → β → α} (h : UniformEquicontinuous F)
    (i : ι) : UniformContinuous (F i) := fun U hU =>
  mem_map.mpr (mem_of_superset (h U hU) fun _ hxy => hxy i)


/-- Each function of a family uniformly equicontinuous on `S` is uniformly continuous on `S`. -/
theorem UniformEquicontinuousOn.uniformContinuousOn {F : ι → β → α} {S : Set β}
    (h : UniformEquicontinuousOn F S) (i : ι) :
    UniformContinuousOn (F i) S := fun U hU =>
  mem_map.mpr (mem_of_superset (h U hU) fun _ hxy => hxy i)


protected theorem Set.UniformEquicontinuous.uniformContinuous_of_mem {H : Set <| β → α}
    (h : H.UniformEquicontinuous) {f : β → α} (hf : f ∈ H) : UniformContinuous f :=
  h.uniformContinuous ⟨f, hf⟩


protected theorem Set.UniformEquicontinuousOn.uniformContinuousOn_of_mem {H : Set <| β → α}
    {S : Set β} (h : H.UniformEquicontinuousOn S) {f : β → α} (hf : f ∈ H) :
    UniformContinuousOn f S :=
  h.uniformContinuousOn ⟨f, hf⟩


/-- Taking sub-families preserves equicontinuity at a point. -/
theorem EquicontinuousAt.comp {F : ι → X → α} {x₀ : X} (h : EquicontinuousAt F x₀) (u : κ → ι) :
    EquicontinuousAt (F ∘ u) x₀ := fun U hU => (h U hU).mono fun _ H k => H (u k)


/-- Taking sub-families preserves equicontinuity at a point within a subset. -/
theorem EquicontinuousWithinAt.comp {F : ι → X → α} {S : Set X} {x₀ : X}
    (h : EquicontinuousWithinAt F S x₀) (u : κ → ι) :
    EquicontinuousWithinAt (F ∘ u) S x₀ :=
  fun U hU ↦ (h U hU).mono fun _ H k => H (u k)


protected theorem Set.EquicontinuousAt.mono {H H' : Set <| X → α} {x₀ : X}
    (h : H.EquicontinuousAt x₀) (hH : H' ⊆ H) : H'.EquicontinuousAt x₀ :=
  h.comp (inclusion hH)


protected theorem Set.EquicontinuousWithinAt.mono {H H' : Set <| X → α} {S : Set X} {x₀ : X}
    (h : H.EquicontinuousWithinAt S x₀) (hH : H' ⊆ H) : H'.EquicontinuousWithinAt S x₀ :=
  h.comp (inclusion hH)


/-- Taking sub-families preserves equicontinuity. -/
theorem Equicontinuous.comp {F : ι → X → α} (h : Equicontinuous F) (u : κ → ι) :
    Equicontinuous (F ∘ u) := fun x => (h x).comp u


/-- Taking sub-families preserves equicontinuity on a subset. -/
theorem EquicontinuousOn.comp {F : ι → X → α} {S : Set X} (h : EquicontinuousOn F S) (u : κ → ι) :
    EquicontinuousOn (F ∘ u) S := fun x hx ↦ (h x hx).comp u


protected theorem Set.Equicontinuous.mono {H H' : Set <| X → α} (h : H.Equicontinuous)
    (hH : H' ⊆ H) : H'.Equicontinuous :=
  h.comp (inclusion hH)


protected theorem Set.EquicontinuousOn.mono {H H' : Set <| X → α} {S : Set X}
    (h : H.EquicontinuousOn S) (hH : H' ⊆ H) : H'.EquicontinuousOn S :=
  h.comp (inclusion hH)


/-- Taking sub-families preserves uniform equicontinuity. -/
theorem UniformEquicontinuous.comp {F : ι → β → α} (h : UniformEquicontinuous F) (u : κ → ι) :
    UniformEquicontinuous (F ∘ u) := fun U hU => (h U hU).mono fun _ H k => H (u k)


/-- Taking sub-families preserves uniform equicontinuity on a subset. -/
theorem UniformEquicontinuousOn.comp {F : ι → β → α} {S : Set β} (h : UniformEquicontinuousOn F S)
    (u : κ → ι) : UniformEquicontinuousOn (F ∘ u) S :=
  fun U hU ↦ (h U hU).mono fun _ H k => H (u k)


protected theorem Set.UniformEquicontinuous.mono {H H' : Set <| β → α} (h : H.UniformEquicontinuous)
    (hH : H' ⊆ H) : H'.UniformEquicontinuous :=
  h.comp (inclusion hH)


protected theorem Set.UniformEquicontinuousOn.mono {H H' : Set <| β → α} {S : Set β}
    (h : H.UniformEquicontinuousOn S) (hH : H' ⊆ H) : H'.UniformEquicontinuousOn S :=
  h.comp (inclusion hH)


/-- A family `𝓕 : ι → X → α` is equicontinuous at `x₀` iff `range 𝓕` is equicontinuous at `x₀`,
i.e the family `(↑) : range F → X → α` is equicontinuous at `x₀`. -/
theorem equicontinuousAt_iff_range {F : ι → X → α} {x₀ : X} :
    EquicontinuousAt F x₀ ↔ EquicontinuousAt ((↑) : range F → X → α) x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    x₀ : X
    ⊢ Iff (EquicontinuousAt F x₀) (EquicontinuousAt Subtype.val x₀)
  -/
  simp only [EquicontinuousAt, forall_subtype_range_iff]
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → X → α` is equicontinuous at `x₀` within `S` iff `range 𝓕` is equicontinuous
at `x₀` within `S`, i.e the family `(↑) : range F → X → α` is equicontinuous at `x₀` within `S`. -/
theorem equicontinuousWithinAt_iff_range {F : ι → X → α} {S : Set X} {x₀ : X} :
    EquicontinuousWithinAt F S x₀ ↔ EquicontinuousWithinAt ((↑) : range F → X → α) S x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    S : Set X
    x₀ : X
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (EquicontinuousWithinAt Subtype.val S x₀)
  -/
  simp only [EquicontinuousWithinAt, forall_subtype_range_iff]
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → X → α` is equicontinuous iff `range 𝓕` is equicontinuous,
i.e the family `(↑) : range F → X → α` is equicontinuous. -/
theorem equicontinuous_iff_range {F : ι → X → α} :
    Equicontinuous F ↔ Equicontinuous ((↑) : range F → X → α) :=
  forall_congr' fun _ => equicontinuousAt_iff_range


/-- A family `𝓕 : ι → X → α` is equicontinuous on `S` iff `range 𝓕` is equicontinuous on `S`,
i.e the family `(↑) : range F → X → α` is equicontinuous on `S`. -/
theorem equicontinuousOn_iff_range {F : ι → X → α} {S : Set X} :
    EquicontinuousOn F S ↔ EquicontinuousOn ((↑) : range F → X → α) S :=
  forall_congr' fun _ ↦ forall_congr' fun _ ↦ equicontinuousWithinAt_iff_range


/-- A family `𝓕 : ι → β → α` is uniformly equicontinuous iff `range 𝓕` is uniformly equicontinuous,
i.e the family `(↑) : range F → β → α` is uniformly equicontinuous. -/
theorem uniformEquicontinuous_iff_range {F : ι → β → α} :
    UniformEquicontinuous F ↔ UniformEquicontinuous ((↑) : range F → β → α) :=
               /-
                 ι : Type u_1
                 α : Type u_6
                 β : Type u_8
                 uα : UniformSpace α
                 uβ : UniformSpace β
                 F : ι → β → α
                 h : UniformEquicontinuous F
                 ⊢ UniformEquicontinuous Subtype.val
               -/
  ⟨fun h => by rw [← comp_rangeSplitting F]; exact h.comp _, fun h =>
                                             /-
                                               🎉 no goals
                                             -/
    h.comp (rangeFactorization F)⟩


/-- A family `𝓕 : ι → β → α` is uniformly equicontinuous on `S` iff `range 𝓕` is uniformly
equicontinuous on `S`, i.e the family `(↑) : range F → β → α` is uniformly equicontinuous on `S`. -/
theorem uniformEquicontinuousOn_iff_range {F : ι → β → α} {S : Set β} :
    UniformEquicontinuousOn F S ↔ UniformEquicontinuousOn ((↑) : range F → β → α) S :=
               /-
                 ι : Type u_1
                 α : Type u_6
                 β : Type u_8
                 uα : UniformSpace α
                 uβ : UniformSpace β
                 F : ι → β → α
                 S : Set β
                 h : UniformEquicontinuousOn F S
                 ⊢ UniformEquicontinuousOn Subtype.val S
               -/
  ⟨fun h => by rw [← comp_rangeSplitting F]; exact h.comp _, fun h =>
                                             /-
                                               🎉 no goals
                                             -/
    h.comp (rangeFactorization F)⟩


/-- A family `𝓕 : ι → X → α` is equicontinuous at `x₀` iff the function `swap 𝓕 : X → ι → α` is
continuous at `x₀` *when `ι → α` is equipped with the topology of uniform convergence*. This is
very useful for developing the equicontinuity API, but it should not be used directly for other
purposes. -/
theorem equicontinuousAt_iff_continuousAt {F : ι → X → α} {x₀ : X} :
    EquicontinuousAt F x₀ ↔ ContinuousAt (ofFun ∘ Function.swap F : X → ι →ᵤ α) x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    x₀ : X
    ⊢ Iff (EquicontinuousAt F x₀) (ContinuousAt (Function.comp (⇑UniformFun.ofFun) …
  -/
  rw [ContinuousAt, (UniformFun.hasBasis_nhds ι α _).tendsto_right_iff]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    x₀ : X
    ⊢ Iff (EquicontinuousAt F x₀) (∀ (i : Set (Prod α α)), Membership.mem (uniform …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → X → α` is equicontinuous at `x₀` within `S` iff the function
`swap 𝓕 : X → ι → α` is continuous at `x₀` within `S`
*when `ι → α` is equipped with the topology of uniform convergence*. This is very useful for
developing the equicontinuity API, but it should not be used directly for other purposes. -/
theorem equicontinuousWithinAt_iff_continuousWithinAt {F : ι → X → α} {S : Set X} {x₀ : X} :
    EquicontinuousWithinAt F S x₀ ↔
    ContinuousWithinAt (ofFun ∘ Function.swap F : X → ι →ᵤ α) S x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    S : Set X
    x₀ : X
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (ContinuousWithinAt (Function.comp (⇑Uni …
  -/
  rw [ContinuousWithinAt, (UniformFun.hasBasis_nhds ι α _).tendsto_right_iff]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    S : Set X
    x₀ : X
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (∀ (i : Set (Prod α α)), Membership.mem  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → X → α` is equicontinuous iff the function `swap 𝓕 : X → ι → α` is
continuous *when `ι → α` is equipped with the topology of uniform convergence*. This is
very useful for developing the equicontinuity API, but it should not be used directly for other
purposes. -/
theorem equicontinuous_iff_continuous {F : ι → X → α} :
    Equicontinuous F ↔ Continuous (ofFun ∘ Function.swap F : X → ι →ᵤ α) := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    ⊢ Iff (Equicontinuous F) (Continuous (Function.comp (⇑UniformFun.ofFun) (Funct …
  -/
  simp_rw [Equicontinuous, continuous_iff_continuousAt, equicontinuousAt_iff_continuousAt]
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → X → α` is equicontinuous on `S` iff the function `swap 𝓕 : X → ι → α` is
continuous on `S` *when `ι → α` is equipped with the topology of uniform convergence*. This is
very useful for developing the equicontinuity API, but it should not be used directly for other
purposes. -/
theorem equicontinuousOn_iff_continuousOn {F : ι → X → α} {S : Set X} :
    EquicontinuousOn F S ↔ ContinuousOn (ofFun ∘ Function.swap F : X → ι →ᵤ α) S := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    F : ι → X → α
    S : Set X
    ⊢ Iff (EquicontinuousOn F S) (ContinuousOn (Function.comp (⇑UniformFun.ofFun)  …
  -/
  simp_rw [EquicontinuousOn, ContinuousOn, equicontinuousWithinAt_iff_continuousWithinAt]
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → β → α` is uniformly equicontinuous iff the function `swap 𝓕 : β → ι → α` is
uniformly continuous *when `ι → α` is equipped with the uniform structure of uniform convergence*.
This is very useful for developing the equicontinuity API, but it should not be used directly
for other purposes. -/
theorem uniformEquicontinuous_iff_uniformContinuous {F : ι → β → α} :
    UniformEquicontinuous F ↔ UniformContinuous (ofFun ∘ Function.swap F : β → ι →ᵤ α) := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    ⊢ Iff (UniformEquicontinuous F) (UniformContinuous (Function.comp (⇑UniformFun …
  -/
  rw [UniformContinuous, (UniformFun.hasBasis_uniformity ι α).tendsto_right_iff]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    ⊢ Iff (UniformEquicontinuous F) (∀ (i : Set (Prod α α)), Membership.mem (unifo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A family `𝓕 : ι → β → α` is uniformly equicontinuous on `S` iff the function
`swap 𝓕 : β → ι → α` is uniformly continuous on `S`
*when `ι → α` is equipped with the uniform structure of uniform convergence*. This is very useful
for developing the equicontinuity API, but it should not be used directly for other purposes. -/
theorem uniformEquicontinuousOn_iff_uniformContinuousOn {F : ι → β → α} {S : Set β} :
    UniformEquicontinuousOn F S ↔ UniformContinuousOn (ofFun ∘ Function.swap F : β → ι →ᵤ α) S := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    S : Set β
    ⊢ Iff (UniformEquicontinuousOn F S) (UniformContinuousOn (Function.comp (⇑Unif …
  -/
  rw [UniformContinuousOn, (UniformFun.hasBasis_uniformity ι α).tendsto_right_iff]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → β → α
    S : Set β
    ⊢ Iff (UniformEquicontinuousOn F S) (∀ (i : Set (Prod α α)), Membership.mem (u …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem equicontinuousWithinAt_iInf_rng {u : κ → UniformSpace α'} {F : ι → X → α'}
    {S : Set X} {x₀ : X} : EquicontinuousWithinAt (uα :=  ⨅ k, u k) F S x₀ ↔
      ∀ k, EquicontinuousWithinAt (uα :=  u k) F S x₀ := by
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    S : Set X
    x₀ : X
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (∀ (k : κ), EquicontinuousWithinAt F S x₀)
  -/
  simp only [equicontinuousWithinAt_iff_continuousWithinAt (uα := _), topologicalSpace]
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    S : Set X
    x₀ : X
    ⊢ Iff (ContinuousWithinAt (Function.comp (⇑UniformFun.ofFun) (Function.swap F) …
  -/
  unfold ContinuousWithinAt
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    S : Set X
    x₀ : X
    ⊢ Iff (Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) (n …
  -/
  rw [UniformFun.iInf_eq, toTopologicalSpace_iInf, nhds_iInf, tendsto_iInf]
  /-
    🎉 no goals
  -/


theorem equicontinuousAt_iInf_rng {u : κ → UniformSpace α'} {F : ι → X → α'}
    {x₀ : X} :
    EquicontinuousAt (uα := ⨅ k, u k) F x₀ ↔ ∀ k, EquicontinuousAt (uα := u k) F x₀ := by
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    x₀ : X
    ⊢ Iff (EquicontinuousAt F x₀) (∀ (k : κ), EquicontinuousAt F x₀)
  -/
  simp only [← equicontinuousWithinAt_univ (uα := _), equicontinuousWithinAt_iInf_rng]
  /-
    🎉 no goals
  -/


theorem equicontinuous_iInf_rng {u : κ → UniformSpace α'} {F : ι → X → α'} :
    Equicontinuous (uα := ⨅ k, u k) F ↔ ∀ k, Equicontinuous (uα := u k) F := by
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    ⊢ Iff (Equicontinuous F) (∀ (k : κ), Equicontinuous F)
  -/
  simp_rw [equicontinuous_iff_continuous (uα := _), UniformFun.topologicalSpace]
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    ⊢ Iff (Continuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F))) (∀ (k …
  -/
  rw [UniformFun.iInf_eq, toTopologicalSpace_iInf, continuous_iInf_rng]
  /-
    🎉 no goals
  -/


theorem equicontinuousOn_iInf_rng {u : κ → UniformSpace α'} {F : ι → X → α'}
    {S : Set X} :
    EquicontinuousOn (uα := ⨅ k, u k) F S ↔ ∀ k, EquicontinuousOn (uα := u k) F S := by
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α' : Type u_7
    tX : TopologicalSpace X
    u : κ → UniformSpace α'
    F : ι → X → α'
    S : Set X
    ⊢ Iff (EquicontinuousOn F S) (∀ (k : κ), EquicontinuousOn F S)
  -/
  simp_rw [EquicontinuousOn, equicontinuousWithinAt_iInf_rng, @forall_swap _ κ]
  /-
    🎉 no goals
  -/


theorem uniformEquicontinuous_iInf_rng {u : κ → UniformSpace α'} {F : ι → β → α'} :
    UniformEquicontinuous (uα := ⨅ k, u k) F ↔ ∀ k, UniformEquicontinuous (uα := u k) F := by
  /-
    ι : Type u_1
    κ : Type u_2
    α' : Type u_7
    β : Type u_8
    uβ : UniformSpace β
    u : κ → UniformSpace α'
    F : ι → β → α'
    ⊢ Iff (UniformEquicontinuous F) (∀ (k : κ), UniformEquicontinuous F)
  -/
  simp_rw [uniformEquicontinuous_iff_uniformContinuous (uα := _)]
  /-
    ι : Type u_1
    κ : Type u_2
    α' : Type u_7
    β : Type u_8
    uβ : UniformSpace β
    u : κ → UniformSpace α'
    F : ι → β → α'
    ⊢ Iff (UniformContinuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) …
  -/
  rw [UniformFun.iInf_eq, uniformContinuous_iInf_rng]
  /-
    🎉 no goals
  -/


theorem uniformEquicontinuousOn_iInf_rng {u : κ → UniformSpace α'} {F : ι → β → α'}
    {S : Set β} : UniformEquicontinuousOn (uα := ⨅ k, u k) F S ↔
      ∀ k, UniformEquicontinuousOn (uα := u k) F S := by
  /-
    ι : Type u_1
    κ : Type u_2
    α' : Type u_7
    β : Type u_8
    uβ : UniformSpace β
    u : κ → UniformSpace α'
    F : ι → β → α'
    S : Set β
    ⊢ Iff (UniformEquicontinuousOn F S) (∀ (k : κ), UniformEquicontinuousOn F S)
  -/
  simp_rw [uniformEquicontinuousOn_iff_uniformContinuousOn (uα := _)]
  /-
    ι : Type u_1
    κ : Type u_2
    α' : Type u_7
    β : Type u_8
    uβ : UniformSpace β
    u : κ → UniformSpace α'
    F : ι → β → α'
    S : Set β
    ⊢ Iff (UniformContinuousOn (Function.comp (⇑UniformFun.ofFun) (Function.swap F …
  -/
  unfold UniformContinuousOn
  /-
    ι : Type u_1
    κ : Type u_2
    α' : Type u_7
    β : Type u_8
    uβ : UniformSpace β
    u : κ → UniformSpace α'
    F : ι → β → α'
    S : Set β
    ⊢ Iff (Filter.Tendsto (fun x => { fst := Function.comp (⇑UniformFun.ofFun) (Fu …
  -/
  rw [UniformFun.iInf_eq, iInf_uniformity, tendsto_iInf]
  /-
    🎉 no goals
  -/


theorem equicontinuousWithinAt_iInf_dom {t : κ → TopologicalSpace X'} {F : ι → X' → α}
    {S : Set X'} {x₀ : X'} {k : κ} (hk : EquicontinuousWithinAt (tX := t k) F S x₀) :
    EquicontinuousWithinAt (tX := ⨅ k, t k) F S x₀ := by
  /-
    ι : Type u_1
    κ : Type u_2
    X' : Type u_4
    α : Type u_6
    uα : UniformSpace α
    t : κ → TopologicalSpace X'
    F : ι → X' → α
    S : Set X'
    x₀ : X'
    k : κ
    hk : EquicontinuousWithinAt F S x₀
    ⊢ EquicontinuousWithinAt F S x₀
  -/
  simp only [equicontinuousWithinAt_iff_continuousWithinAt (tX := _)] at hk ⊢
  /-
    ι : Type u_1
    κ : Type u_2
    X' : Type u_4
    α : Type u_6
    uα : UniformSpace α
    t : κ → TopologicalSpace X'
    F : ι → X' → α
    S : Set X'
    x₀ : X'
    k : κ
    hk : ContinuousWithinAt (Function.comp (⇑UniformFun.ofFun) (Function.swap F))  …
    ⊢ ContinuousWithinAt (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) S x₀
  -/
  unfold ContinuousWithinAt nhdsWithin at hk ⊢
  /-
    ι : Type u_1
    κ : Type u_2
    X' : Type u_4
    α : Type u_6
    uα : UniformSpace α
    t : κ → TopologicalSpace X'
    F : ι → X' → α
    S : Set X'
    x₀ : X'
    k : κ
    hk : Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) (Min …
    ⊢ Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) (Min.mi …
  -/
  rw [nhds_iInf]
  /-
    ι : Type u_1
    κ : Type u_2
    X' : Type u_4
    α : Type u_6
    uα : UniformSpace α
    t : κ → TopologicalSpace X'
    F : ι → X' → α
    S : Set X'
    x₀ : X'
    k : κ
    hk : Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) (Min …
    ⊢ Filter.Tendsto (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) (Min.mi …
  -/
  exact hk.mono_left <| inf_le_inf_right _ <| iInf_le _ k
  /-
    🎉 no goals
  -/


theorem equicontinuousAt_iInf_dom {t : κ → TopologicalSpace X'} {F : ι → X' → α}
    {x₀ : X'} {k : κ} (hk : EquicontinuousAt (tX := t k) F x₀) :
    EquicontinuousAt (tX := ⨅ k, t k) F x₀ := by
  /-
    ι : Type u_1
    κ : Type u_2
    X' : Type u_4
    α : Type u_6
    uα : UniformSpace α
    t : κ → TopologicalSpace X'
    F : ι → X' → α
    x₀ : X'
    k : κ
    hk : EquicontinuousAt F x₀
    ⊢ EquicontinuousAt F x₀
  -/
  rw [← equicontinuousWithinAt_univ (tX := _)] at hk ⊢
  /-
    ι : Type u_1
    κ : Type u_2
    X' : Type u_4
    α : Type u_6
    uα : UniformSpace α
    t : κ → TopologicalSpace X'
    F : ι → X' → α
    x₀ : X'
    k : κ
    hk : EquicontinuousWithinAt F Set.univ x₀
    ⊢ EquicontinuousWithinAt F Set.univ x₀
  -/
  exact equicontinuousWithinAt_iInf_dom hk
  /-
    🎉 no goals
  -/


theorem equicontinuous_iInf_dom {t : κ → TopologicalSpace X'} {F : ι → X' → α}
    {k : κ} (hk : Equicontinuous (tX := t k) F) :
    Equicontinuous (tX := ⨅ k, t k) F :=
  fun x ↦ equicontinuousAt_iInf_dom (hk x)


theorem equicontinuousOn_iInf_dom {t : κ → TopologicalSpace X'} {F : ι → X' → α}
    {S : Set X'} {k : κ} (hk : EquicontinuousOn (tX := t k) F S) :
    EquicontinuousOn (tX := ⨅ k, t k) F S :=
  fun x hx ↦ equicontinuousWithinAt_iInf_dom (hk x hx)


theorem uniformEquicontinuous_iInf_dom {u : κ → UniformSpace β'} {F : ι → β' → α}
    {k : κ} (hk : UniformEquicontinuous (uβ := u k) F) :
    UniformEquicontinuous (uβ := ⨅ k, u k) F := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β' : Type u_9
    uα : UniformSpace α
    u : κ → UniformSpace β'
    F : ι → β' → α
    k : κ
    hk : UniformEquicontinuous F
    ⊢ UniformEquicontinuous F
  -/
  simp_rw [uniformEquicontinuous_iff_uniformContinuous (uβ := _)] at hk ⊢
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β' : Type u_9
    uα : UniformSpace α
    u : κ → UniformSpace β'
    F : ι → β' → α
    k : κ
    hk : UniformContinuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F))
    ⊢ UniformContinuous (Function.comp (⇑UniformFun.ofFun) (Function.swap F))
  -/
  exact uniformContinuous_iInf_dom hk
  /-
    🎉 no goals
  -/


theorem uniformEquicontinuousOn_iInf_dom {u : κ → UniformSpace β'} {F : ι → β' → α}
    {S : Set β'} {k : κ} (hk : UniformEquicontinuousOn (uβ := u k) F S) :
    UniformEquicontinuousOn (uβ := ⨅ k, u k) F S := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β' : Type u_9
    uα : UniformSpace α
    u : κ → UniformSpace β'
    F : ι → β' → α
    S : Set β'
    k : κ
    hk : UniformEquicontinuousOn F S
    ⊢ UniformEquicontinuousOn F S
  -/
  simp_rw [uniformEquicontinuousOn_iff_uniformContinuousOn (uβ := _)] at hk ⊢
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β' : Type u_9
    uα : UniformSpace α
    u : κ → UniformSpace β'
    F : ι → β' → α
    S : Set β'
    k : κ
    hk : UniformContinuousOn (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) S
    ⊢ UniformContinuousOn (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) S
  -/
  unfold UniformContinuousOn
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β' : Type u_9
    uα : UniformSpace α
    u : κ → UniformSpace β'
    F : ι → β' → α
    S : Set β'
    k : κ
    hk : UniformContinuousOn (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) S
    ⊢ Filter.Tendsto (fun x => { fst := Function.comp (⇑UniformFun.ofFun) (Functio …
  -/
  rw [iInf_uniformity]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β' : Type u_9
    uα : UniformSpace α
    u : κ → UniformSpace β'
    F : ι → β' → α
    S : Set β'
    k : κ
    hk : UniformContinuousOn (Function.comp (⇑UniformFun.ofFun) (Function.swap F)) S
    ⊢ Filter.Tendsto (fun x => { fst := Function.comp (⇑UniformFun.ofFun) (Functio …
  -/
  exact hk.mono_left <| inf_le_inf_right _ <| iInf_le _ k
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.equicontinuousAt_iff_left {p : κ → Prop} {s : κ → Set X}
    {F : ι → X → α} {x₀ : X} (hX : (𝓝 x₀).HasBasis p s) :
    EquicontinuousAt F x₀ ↔ ∀ U ∈ 𝓤 α, ∃ k, p k ∧ ∀ x ∈ s k, ∀ i, (F i x₀, F i x) ∈ U := by
  rw [equicontinuousAt_iff_continuousAt, ContinuousAt,
    hX.tendsto_iff (UniformFun.hasBasis_nhds ι α _)]
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    p : κ → Prop
    s : κ → Set X
    F : ι → X → α
    x₀ : X
    hX : (nhds x₀).HasBasis p s
    ⊢ Iff (∀ (ib : Set (Prod α α)), Membership.mem (uniformity α) ib → Exists fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.equicontinuousWithinAt_iff_left {p : κ → Prop} {s : κ → Set X}
    {F : ι → X → α} {S : Set X} {x₀ : X} (hX : (𝓝[S] x₀).HasBasis p s) :
    EquicontinuousWithinAt F S x₀ ↔ ∀ U ∈ 𝓤 α, ∃ k, p k ∧ ∀ x ∈ s k, ∀ i, (F i x₀, F i x) ∈ U := by
  rw [equicontinuousWithinAt_iff_continuousWithinAt, ContinuousWithinAt,
    hX.tendsto_iff (UniformFun.hasBasis_nhds ι α _)]
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    p : κ → Prop
    s : κ → Set X
    F : ι → X → α
    S : Set X
    x₀ : X
    hX : (nhdsWithin x₀ S).HasBasis p s
    ⊢ Iff (∀ (ib : Set (Prod α α)), Membership.mem (uniformity α) ib → Exists fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.equicontinuousAt_iff_right {p : κ → Prop} {s : κ → Set (α × α)}
    {F : ι → X → α} {x₀ : X} (hα : (𝓤 α).HasBasis p s) :
    EquicontinuousAt F x₀ ↔ ∀ k, p k → ∀ᶠ x in 𝓝 x₀, ∀ i, (F i x₀, F i x) ∈ s k := by
  rw [equicontinuousAt_iff_continuousAt, ContinuousAt,
    (UniformFun.hasBasis_nhds_of_basis ι α _ hα).tendsto_right_iff]
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    p : κ → Prop
    s : κ → Set (Prod α α)
    F : ι → X → α
    x₀ : X
    hα : (uniformity α).HasBasis p s
    ⊢ Iff (∀ (i : κ), p i → Filter.Eventually (fun x => Membership.mem (setOf fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.equicontinuousWithinAt_iff_right {p : κ → Prop}
    {s : κ → Set (α × α)} {F : ι → X → α} {S : Set X} {x₀ : X} (hα : (𝓤 α).HasBasis p s) :
    EquicontinuousWithinAt F S x₀ ↔ ∀ k, p k → ∀ᶠ x in 𝓝[S] x₀, ∀ i, (F i x₀, F i x) ∈ s k := by
  rw [equicontinuousWithinAt_iff_continuousWithinAt, ContinuousWithinAt,
    (UniformFun.hasBasis_nhds_of_basis ι α _ hα).tendsto_right_iff]
  /-
    ι : Type u_1
    κ : Type u_2
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    p : κ → Prop
    s : κ → Set (Prod α α)
    F : ι → X → α
    S : Set X
    x₀ : X
    hα : (uniformity α).HasBasis p s
    ⊢ Iff (∀ (i : κ), p i → Filter.Eventually (fun x => Membership.mem (setOf fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.equicontinuousAt_iff {κ₁ κ₂ : Type*} {p₁ : κ₁ → Prop} {s₁ : κ₁ → Set X}
    {p₂ : κ₂ → Prop} {s₂ : κ₂ → Set (α × α)} {F : ι → X → α} {x₀ : X} (hX : (𝓝 x₀).HasBasis p₁ s₁)
    (hα : (𝓤 α).HasBasis p₂ s₂) :
    EquicontinuousAt F x₀ ↔
      ∀ k₂, p₂ k₂ → ∃ k₁, p₁ k₁ ∧ ∀ x ∈ s₁ k₁, ∀ i, (F i x₀, F i x) ∈ s₂ k₂ := by
  rw [equicontinuousAt_iff_continuousAt, ContinuousAt,
    hX.tendsto_iff (UniformFun.hasBasis_nhds_of_basis ι α _ hα)]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    κ₁ : Type u_11
    κ₂ : Type u_12
    p₁ : κ₁ → Prop
    s₁ : κ₁ → Set X
    p₂ : κ₂ → Prop
    s₂ : κ₂ → Set (Prod α α)
    F : ι → X → α
    x₀ : X
    hX : (nhds x₀).HasBasis p₁ s₁
    hα : (uniformity α).HasBasis p₂ s₂
    ⊢ Iff (∀ (ib : κ₂), p₂ ib → Exists fun ia => And (p₁ ia) (∀ (x : X), Membershi …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.equicontinuousWithinAt_iff {κ₁ κ₂ : Type*} {p₁ : κ₁ → Prop}
    {s₁ : κ₁ → Set X} {p₂ : κ₂ → Prop} {s₂ : κ₂ → Set (α × α)} {F : ι → X → α} {S : Set X} {x₀ : X}
    (hX : (𝓝[S] x₀).HasBasis p₁ s₁) (hα : (𝓤 α).HasBasis p₂ s₂) :
    EquicontinuousWithinAt F S x₀ ↔
      ∀ k₂, p₂ k₂ → ∃ k₁, p₁ k₁ ∧ ∀ x ∈ s₁ k₁, ∀ i, (F i x₀, F i x) ∈ s₂ k₂ := by
  rw [equicontinuousWithinAt_iff_continuousWithinAt, ContinuousWithinAt,
    hX.tendsto_iff (UniformFun.hasBasis_nhds_of_basis ι α _ hα)]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    κ₁ : Type u_11
    κ₂ : Type u_12
    p₁ : κ₁ → Prop
    s₁ : κ₁ → Set X
    p₂ : κ₂ → Prop
    s₂ : κ₂ → Set (Prod α α)
    F : ι → X → α
    S : Set X
    x₀ : X
    hX : (nhdsWithin x₀ S).HasBasis p₁ s₁
    hα : (uniformity α).HasBasis p₂ s₂
    ⊢ Iff (∀ (ib : κ₂), p₂ ib → Exists fun ia => And (p₁ ia) (∀ (x : X), Membershi …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.uniformEquicontinuous_iff_left {p : κ → Prop}
    {s : κ → Set (β × β)} {F : ι → β → α} (hβ : (𝓤 β).HasBasis p s) :
    UniformEquicontinuous F ↔
      ∀ U ∈ 𝓤 α, ∃ k, p k ∧ ∀ x y, (x, y) ∈ s k → ∀ i, (F i x, F i y) ∈ U := by
  rw [uniformEquicontinuous_iff_uniformContinuous, UniformContinuous,
    hβ.tendsto_iff (UniformFun.hasBasis_uniformity ι α)]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    p : κ → Prop
    s : κ → Set (Prod β β)
    F : ι → β → α
    hβ : (uniformity β).HasBasis p s
    ⊢ Iff (∀ (ib : Set (Prod α α)), Membership.mem (uniformity α) ib → Exists fun  …
  -/
  simp only [Prod.forall]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    p : κ → Prop
    s : κ → Set (Prod β β)
    F : ι → β → α
    hβ : (uniformity β).HasBasis p s
    ⊢ Iff (∀ (ib : Set (Prod α α)), Membership.mem (uniformity α) ib → Exists fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.uniformEquicontinuousOn_iff_left {p : κ → Prop}
    {s : κ → Set (β × β)} {F : ι → β → α} {S : Set β} (hβ : (𝓤 β ⊓ 𝓟 (S ×ˢ S)).HasBasis p s) :
    UniformEquicontinuousOn F S ↔
      ∀ U ∈ 𝓤 α, ∃ k, p k ∧ ∀ x y, (x, y) ∈ s k → ∀ i, (F i x, F i y) ∈ U := by
  rw [uniformEquicontinuousOn_iff_uniformContinuousOn, UniformContinuousOn,
    hβ.tendsto_iff (UniformFun.hasBasis_uniformity ι α)]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    p : κ → Prop
    s : κ → Set (Prod β β)
    F : ι → β → α
    S : Set β
    hβ : (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))).HasBasis p s
    ⊢ Iff (∀ (ib : Set (Prod α α)), Membership.mem (uniformity α) ib → Exists fun  …
  -/
  simp only [Prod.forall]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    p : κ → Prop
    s : κ → Set (Prod β β)
    F : ι → β → α
    S : Set β
    hβ : (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))).HasBasis p s
    ⊢ Iff (∀ (ib : Set (Prod α α)), Membership.mem (uniformity α) ib → Exists fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.uniformEquicontinuous_iff_right {p : κ → Prop}
    {s : κ → Set (α × α)} {F : ι → β → α} (hα : (𝓤 α).HasBasis p s) :
    UniformEquicontinuous F ↔ ∀ k, p k → ∀ᶠ xy : β × β in 𝓤 β, ∀ i, (F i xy.1, F i xy.2) ∈ s k := by
  rw [uniformEquicontinuous_iff_uniformContinuous, UniformContinuous,
    (UniformFun.hasBasis_uniformity_of_basis ι α hα).tendsto_right_iff]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    p : κ → Prop
    s : κ → Set (Prod α α)
    F : ι → β → α
    hα : (uniformity α).HasBasis p s
    ⊢ Iff (∀ (i : κ), p i → Filter.Eventually (fun x => Membership.mem (Function.c …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.uniformEquicontinuousOn_iff_right {p : κ → Prop}
    {s : κ → Set (α × α)} {F : ι → β → α} {S : Set β} (hα : (𝓤 α).HasBasis p s) :
    UniformEquicontinuousOn F S ↔
      ∀ k, p k → ∀ᶠ xy : β × β in 𝓤 β ⊓ 𝓟 (S ×ˢ S), ∀ i, (F i xy.1, F i xy.2) ∈ s k := by
  rw [uniformEquicontinuousOn_iff_uniformContinuousOn, UniformContinuousOn,
    (UniformFun.hasBasis_uniformity_of_basis ι α hα).tendsto_right_iff]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    p : κ → Prop
    s : κ → Set (Prod α α)
    F : ι → β → α
    S : Set β
    hα : (uniformity α).HasBasis p s
    ⊢ Iff (∀ (i : κ), p i → Filter.Eventually (fun x => Membership.mem (Function.c …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.uniformEquicontinuous_iff {κ₁ κ₂ : Type*} {p₁ : κ₁ → Prop}
    {s₁ : κ₁ → Set (β × β)} {p₂ : κ₂ → Prop} {s₂ : κ₂ → Set (α × α)} {F : ι → β → α}
    (hβ : (𝓤 β).HasBasis p₁ s₁) (hα : (𝓤 α).HasBasis p₂ s₂) :
    UniformEquicontinuous F ↔
      ∀ k₂, p₂ k₂ → ∃ k₁, p₁ k₁ ∧ ∀ x y, (x, y) ∈ s₁ k₁ → ∀ i, (F i x, F i y) ∈ s₂ k₂ := by
  rw [uniformEquicontinuous_iff_uniformContinuous, UniformContinuous,
    hβ.tendsto_iff (UniformFun.hasBasis_uniformity_of_basis ι α hα)]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    κ₁ : Type u_11
    κ₂ : Type u_12
    p₁ : κ₁ → Prop
    s₁ : κ₁ → Set (Prod β β)
    p₂ : κ₂ → Prop
    s₂ : κ₂ → Set (Prod α α)
    F : ι → β → α
    hβ : (uniformity β).HasBasis p₁ s₁
    hα : (uniformity α).HasBasis p₂ s₂
    ⊢ Iff (∀ (ib : κ₂), p₂ ib → Exists fun ia => And (p₁ ia) (∀ (x : Prod β β), Me …
  -/
  simp only [Prod.forall]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    κ₁ : Type u_11
    κ₂ : Type u_12
    p₁ : κ₁ → Prop
    s₁ : κ₁ → Set (Prod β β)
    p₂ : κ₂ → Prop
    s₂ : κ₂ → Set (Prod α α)
    F : ι → β → α
    hβ : (uniformity β).HasBasis p₁ s₁
    hα : (uniformity α).HasBasis p₂ s₂
    ⊢ Iff (∀ (ib : κ₂), p₂ ib → Exists fun ia => And (p₁ ia) (∀ (a b : β), Members …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.uniformEquicontinuousOn_iff {κ₁ κ₂ : Type*} {p₁ : κ₁ → Prop}
    {s₁ : κ₁ → Set (β × β)} {p₂ : κ₂ → Prop} {s₂ : κ₂ → Set (α × α)} {F : ι → β → α}
    {S : Set β} (hβ : (𝓤 β ⊓ 𝓟 (S ×ˢ S)).HasBasis p₁ s₁) (hα : (𝓤 α).HasBasis p₂ s₂) :
    UniformEquicontinuousOn F S ↔
      ∀ k₂, p₂ k₂ → ∃ k₁, p₁ k₁ ∧ ∀ x y, (x, y) ∈ s₁ k₁ → ∀ i, (F i x, F i y) ∈ s₂ k₂ := by
  rw [uniformEquicontinuousOn_iff_uniformContinuousOn, UniformContinuousOn,
    hβ.tendsto_iff (UniformFun.hasBasis_uniformity_of_basis ι α hα)]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    κ₁ : Type u_11
    κ₂ : Type u_12
    p₁ : κ₁ → Prop
    s₁ : κ₁ → Set (Prod β β)
    p₂ : κ₂ → Prop
    s₂ : κ₂ → Set (Prod α α)
    F : ι → β → α
    S : Set β
    hβ : (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))).HasBasis p₁ …
    hα : (uniformity α).HasBasis p₂ s₂
    ⊢ Iff (∀ (ib : κ₂), p₂ ib → Exists fun ia => And (p₁ ia) (∀ (x : Prod β β), Me …
  -/
  simp only [Prod.forall]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    κ₁ : Type u_11
    κ₂ : Type u_12
    p₁ : κ₁ → Prop
    s₁ : κ₁ → Set (Prod β β)
    p₂ : κ₂ → Prop
    s₂ : κ₂ → Set (Prod α α)
    F : ι → β → α
    S : Set β
    hβ : (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))).HasBasis p₁ …
    hα : (uniformity α).HasBasis p₂ s₂
    ⊢ Iff (∀ (ib : κ₂), p₂ ib → Exists fun ia => And (p₁ ia) (∀ (a b : β), Members …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given `u : α → β` a uniform inducing map, a family `𝓕 : ι → X → α` is equicontinuous at a point
`x₀ : X` iff the family `𝓕'`, obtained by composing each function of `𝓕` by `u`, is
equicontinuous at `x₀`. -/
theorem IsUniformInducing.equicontinuousAt_iff {F : ι → X → α} {x₀ : X} {u : α → β}
    (hu : IsUniformInducing u) : EquicontinuousAt F x₀ ↔ EquicontinuousAt ((u ∘ ·) ∘ F) x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    x₀ : X
    u : α → β
    hu : IsUniformInducing u
    ⊢ Iff (EquicontinuousAt F x₀) (EquicontinuousAt (Function.comp (fun x => Funct …
  -/
  have := (UniformFun.postcomp_isUniformInducing (α := ι) hu).isInducing
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    x₀ : X
    u : α → β
    hu : IsUniformInducing u
    this : Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp ( …
    ⊢ Iff (EquicontinuousAt F x₀) (EquicontinuousAt (Function.comp (fun x => Funct …
  -/
  rw [equicontinuousAt_iff_continuousAt, equicontinuousAt_iff_continuousAt, this.continuousAt_iff]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    x₀ : X
    u : α → β
    hu : IsUniformInducing u
    this : Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp ( …
    ⊢ Iff (ContinuousAt (Function.comp (Function.comp (⇑UniformFun.ofFun) (Functio …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.equicontinuousAt_iff := IsUniformInducing.equicontinuousAt_iff


/-- Given `u : α → β` a uniform inducing map, a family `𝓕 : ι → X → α` is equicontinuous at a point
`x₀ : X` within a subset `S : Set X` iff the family `𝓕'`, obtained by composing each function
of `𝓕` by `u`, is equicontinuous at `x₀` within `S`. -/
lemma IsUniformInducing.equicontinuousWithinAt_iff {F : ι → X → α} {S : Set X} {x₀ : X} {u : α → β}
    (hu : IsUniformInducing u) : EquicontinuousWithinAt F S x₀ ↔
      EquicontinuousWithinAt ((u ∘ ·) ∘ F) S x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    S : Set X
    x₀ : X
    u : α → β
    hu : IsUniformInducing u
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (EquicontinuousWithinAt (Function.comp ( …
  -/
  have := (UniformFun.postcomp_isUniformInducing (α := ι) hu).isInducing
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    S : Set X
    x₀ : X
    u : α → β
    hu : IsUniformInducing u
    this : Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp ( …
    ⊢ Iff (EquicontinuousWithinAt F S x₀) (EquicontinuousWithinAt (Function.comp ( …
  -/
  simp only [equicontinuousWithinAt_iff_continuousWithinAt, this.continuousWithinAt_iff]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    S : Set X
    x₀ : X
    u : α → β
    hu : IsUniformInducing u
    this : Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp ( …
    ⊢ Iff (ContinuousWithinAt (Function.comp (Function.comp (⇑UniformFun.ofFun) (F …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.equicontinuousWithinAt_iff := IsUniformInducing.equicontinuousWithinAt_iff


/-- Given `u : α → β` a uniform inducing map, a family `𝓕 : ι → X → α` is equicontinuous iff the
family `𝓕'`, obtained by composing each function of `𝓕` by `u`, is equicontinuous. -/
lemma IsUniformInducing.equicontinuous_iff {F : ι → X → α} {u : α → β} (hu : IsUniformInducing u) :
    Equicontinuous F ↔ Equicontinuous ((u ∘ ·) ∘ F) := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    u : α → β
    hu : IsUniformInducing u
    ⊢ Iff (Equicontinuous F) (Equicontinuous (Function.comp (fun x => Function.com …
  -/
  congrm ∀ x, ?_
  /-
    case a
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    u : α → β
    hu : IsUniformInducing u
    x : X
    ⊢ Iff (EquicontinuousAt F x) (EquicontinuousAt (Function.comp (fun x => Functi …
  -/
  rw [hu.equicontinuousAt_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.equicontinuous_iff := IsUniformInducing.equicontinuous_iff


/-- Given `u : α → β` a uniform inducing map, a family `𝓕 : ι → X → α` is equicontinuous on a
subset `S : Set X` iff the family `𝓕'`, obtained by composing each function of `𝓕` by `u`, is
equicontinuous on `S`. -/
theorem IsUniformInducing.equicontinuousOn_iff {F : ι → X → α} {S : Set X} {u : α → β}
    (hu : IsUniformInducing u) : EquicontinuousOn F S ↔ EquicontinuousOn ((u ∘ ·) ∘ F) S := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    S : Set X
    u : α → β
    hu : IsUniformInducing u
    ⊢ Iff (EquicontinuousOn F S) (EquicontinuousOn (Function.comp (fun x => Functi …
  -/
  congrm ∀ x ∈ S, ?_
  /-
    case a
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    β : Type u_8
    tX : TopologicalSpace X
    uα : UniformSpace α
    uβ : UniformSpace β
    F : ι → X → α
    S : Set X
    u : α → β
    hu : IsUniformInducing u
    x : X
    ⊢ Iff (EquicontinuousWithinAt F S x) (EquicontinuousWithinAt (Function.comp (f …
  -/
  rw [hu.equicontinuousWithinAt_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.equicontinuousOn_iff := IsUniformInducing.equicontinuousOn_iff


/-- Given `u : α → γ` a uniform inducing map, a family `𝓕 : ι → β → α` is uniformly equicontinuous
iff the family `𝓕'`, obtained by composing each function of `𝓕` by `u`, is uniformly
equicontinuous. -/
theorem IsUniformInducing.uniformEquicontinuous_iff {F : ι → β → α} {u : α → γ}
    (hu : IsUniformInducing u) : UniformEquicontinuous F ↔ UniformEquicontinuous ((u ∘ ·) ∘ F) := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    γ : Type u_10
    uα : UniformSpace α
    uβ : UniformSpace β
    uγ : UniformSpace γ
    F : ι → β → α
    u : α → γ
    hu : IsUniformInducing u
    ⊢ Iff (UniformEquicontinuous F) (UniformEquicontinuous (Function.comp (fun x = …
  -/
  have := UniformFun.postcomp_isUniformInducing (α := ι) hu
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    γ : Type u_10
    uα : UniformSpace α
    uβ : UniformSpace β
    uγ : UniformSpace γ
    F : ι → β → α
    u : α → γ
    hu : IsUniformInducing u
    this : IsUniformInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp (fu …
    ⊢ Iff (UniformEquicontinuous F) (UniformEquicontinuous (Function.comp (fun x = …
  -/
  simp only [uniformEquicontinuous_iff_uniformContinuous, this.uniformContinuous_iff]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    γ : Type u_10
    uα : UniformSpace α
    uβ : UniformSpace β
    uγ : UniformSpace γ
    F : ι → β → α
    u : α → γ
    hu : IsUniformInducing u
    this : IsUniformInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp (fu …
    ⊢ Iff (UniformContinuous (Function.comp (Function.comp (⇑UniformFun.ofFun) (Fu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformEquicontinuous_iff := IsUniformInducing.uniformEquicontinuous_iff


/-- Given `u : α → γ` a uniform inducing map, a family `𝓕 : ι → β → α` is uniformly equicontinuous
on a subset `S : Set β` iff the family `𝓕'`, obtained by composing each function of `𝓕` by `u`,
is uniformly equicontinuous on `S`. -/
theorem IsUniformInducing.uniformEquicontinuousOn_iff {F : ι → β → α} {S : Set β} {u : α → γ}
    (hu : IsUniformInducing u) :
    UniformEquicontinuousOn F S ↔ UniformEquicontinuousOn ((u ∘ ·) ∘ F) S := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    γ : Type u_10
    uα : UniformSpace α
    uβ : UniformSpace β
    uγ : UniformSpace γ
    F : ι → β → α
    S : Set β
    u : α → γ
    hu : IsUniformInducing u
    ⊢ Iff (UniformEquicontinuousOn F S) (UniformEquicontinuousOn (Function.comp (f …
  -/
  have := UniformFun.postcomp_isUniformInducing (α := ι) hu
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    γ : Type u_10
    uα : UniformSpace α
    uβ : UniformSpace β
    uγ : UniformSpace γ
    F : ι → β → α
    S : Set β
    u : α → γ
    hu : IsUniformInducing u
    this : IsUniformInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp (fu …
    ⊢ Iff (UniformEquicontinuousOn F S) (UniformEquicontinuousOn (Function.comp (f …
  -/
  simp only [uniformEquicontinuousOn_iff_uniformContinuousOn, this.uniformContinuousOn_iff]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    γ : Type u_10
    uα : UniformSpace α
    uβ : UniformSpace β
    uγ : UniformSpace γ
    F : ι → β → α
    S : Set β
    u : α → γ
    hu : IsUniformInducing u
    this : IsUniformInducing (Function.comp (⇑UniformFun.ofFun) (Function.comp (fu …
    ⊢ Iff (UniformContinuousOn (Function.comp (Function.comp (⇑UniformFun.ofFun) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformEquicontinuousOn_iff := IsUniformInducing.uniformEquicontinuousOn_iff


/-- If a set of functions is equicontinuous at some `x₀` within a set `S`, the same is true for its
closure in *any* topology for which evaluation at any `x ∈ S ∪ {x₀}` is continuous. Since
this will be applied to `DFunLike` types, we state it for any topological space with a map
to `X → α` satisfying the right continuity conditions. See also `Set.EquicontinuousWithinAt.closure`
for a more familiar (but weaker) statement.

Note: This could *technically* be called `EquicontinuousWithinAt.closure` without name clashes
with `Set.EquicontinuousWithinAt.closure`, but we don't do it because, even with a `protected`
marker, it would introduce ambiguities while working in namespace `Set` (e.g, in the proof of
any theorem called `Set.something`). -/
theorem EquicontinuousWithinAt.closure' {A : Set Y} {u : Y → X → α} {S : Set X} {x₀ : X}
    (hA : EquicontinuousWithinAt (u ∘ (↑) : A → X → α) S x₀) (hu₁ : Continuous (S.restrict ∘ u))
    (hu₂ : Continuous (eval x₀ ∘ u)) :
    EquicontinuousWithinAt (u ∘ (↑) : closure A → X → α) S x₀ := by
  /-
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    ⊢ EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
  -/
  intro U hU
  /-
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Filter.Eventually (fun x => ∀ (i : ↑(closure A)), Membership.mem U { fst :=  …
  -/
  rcases mem_uniformity_isClosed hU with ⟨V, hV, hVclosed, hVU⟩
  /-
    case intro.intro.intro
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    ⊢ Filter.Eventually (fun x => ∀ (i : ↑(closure A)), Membership.mem U { fst :=  …
  -/
  filter_upwards [hA V hV, eventually_mem_nhdsWithin] with x hx hxS
  /-
    case h
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x : X
    hx : ∀ (i : ↑A), Membership.mem V { fst := Function.comp u Subtype.val i x₀, s …
    hxS : Membership.mem S x
    ⊢ ∀ (i : ↑(closure A)), Membership.mem U { fst := Function.comp u Subtype.val  …
  -/
  rw [SetCoe.forall] at *
  /-
    case h
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x : X
    hx : ∀ (x_1 : Y) (h : Membership.mem A x_1), Membership.mem V { fst := Functio …
    hxS : Membership.mem S x
    ⊢ ∀ (x_1 : Y) (h : Membership.mem (closure A) x_1), Membership.mem U { fst :=  …
  -/
  change A ⊆ (fun f => (u f x₀, u f x)) ⁻¹' V at hx
  /-
    case h
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x : X
    hxS : Membership.mem S x
    hx : HasSubset.Subset A (Set.preimage (fun f => { fst := u f x₀, snd := u f x  …
    ⊢ ∀ (x_1 : Y) (h : Membership.mem (closure A) x_1), Membership.mem U { fst :=  …
  -/
  refine (closure_minimal hx <| hVclosed.preimage <| hu₂.prod_mk ?_).trans (preimage_mono hVU)
  /-
    case h
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    S : Set X
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) S x₀
    hu₁ : Continuous (Function.comp S.restrict u)
    hu₂ : Continuous (Function.comp (Function.eval x₀) u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x : X
    hxS : Membership.mem S x
    hx : HasSubset.Subset A (Set.preimage (fun f => { fst := u f x₀, snd := u f x  …
    ⊢ Continuous fun f => u f x
  -/
  exact (continuous_apply ⟨x, hxS⟩).comp hu₁
  /-
    🎉 no goals
  -/


/-- If a set of functions is equicontinuous at some `x₀`, the same is true for its closure in *any*
topology for which evaluation at any point is continuous. Since this will be applied to
`DFunLike` types, we state it for any topological space with a map to `X → α` satisfying the right
continuity conditions. See also `Set.EquicontinuousAt.closure` for a more familiar statement. -/
theorem EquicontinuousAt.closure' {A : Set Y} {u : Y → X → α} {x₀ : X}
    (hA : EquicontinuousAt (u ∘ (↑) : A → X → α) x₀) (hu : Continuous u) :
    EquicontinuousAt (u ∘ (↑) : closure A → X → α) x₀ := by
  /-
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    x₀ : X
    hA : EquicontinuousAt (Function.comp u Subtype.val) x₀
    hu : Continuous u
    ⊢ EquicontinuousAt (Function.comp u Subtype.val) x₀
  -/
  rw [← equicontinuousWithinAt_univ] at hA ⊢
  /-
    X : Type u_3
    Y : Type u_5
    α : Type u_6
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    uα : UniformSpace α
    A : Set Y
    u : Y → X → α
    x₀ : X
    hA : EquicontinuousWithinAt (Function.comp u Subtype.val) Set.univ x₀
    hu : Continuous u
    ⊢ EquicontinuousWithinAt (Function.comp u Subtype.val) Set.univ x₀
  -/
  exact hA.closure' (Pi.continuous_restrict _ |>.comp hu) (continuous_apply x₀ |>.comp hu)
  /-
    🎉 no goals
  -/


/-- If a set of functions is equicontinuous at some `x₀`, its closure for the product topology is
also equicontinuous at `x₀`. -/
protected theorem Set.EquicontinuousAt.closure {A : Set (X → α)} {x₀ : X}
    (hA : A.EquicontinuousAt x₀) : (closure A).EquicontinuousAt x₀ :=
  hA.closure' (u := id) continuous_id


/-- If a set of functions is equicontinuous at some `x₀` within a set `S`, its closure for the
product topology is also equicontinuous at `x₀` within `S`. This would also be true for the coarser
topology of pointwise convergence on `S ∪ {x₀}`, see `Set.EquicontinuousWithinAt.closure'`. -/
protected theorem Set.EquicontinuousWithinAt.closure {A : Set (X → α)} {S : Set X} {x₀ : X}
    (hA : A.EquicontinuousWithinAt S x₀) :
    (closure A).EquicontinuousWithinAt S x₀ :=
  hA.closure' (u := id) (Pi.continuous_restrict _) (continuous_apply _)


/-- If a set of functions is equicontinuous, the same is true for its closure in *any*
topology for which evaluation at any point is continuous. Since this will be applied to
`DFunLike` types, we state it for any topological space with a map to `X → α` satisfying the right
continuity conditions. See also `Set.Equicontinuous.closure` for a more familiar statement. -/
theorem Equicontinuous.closure' {A : Set Y} {u : Y → X → α}
    (hA : Equicontinuous (u ∘ (↑) : A → X → α)) (hu : Continuous u) :
    Equicontinuous (u ∘ (↑) : closure A → X → α) := fun x ↦ (hA x).closure' hu


/-- If a set of functions is equicontinuous on a set `S`, the same is true for its closure in *any*
topology for which evaluation at any `x ∈ S` is continuous. Since this will be applied to
`DFunLike` types, we state it for any topological space with a map to `X → α` satisfying the right
continuity conditions. See also `Set.EquicontinuousOn.closure` for a more familiar
(but weaker) statement. -/
theorem EquicontinuousOn.closure' {A : Set Y} {u : Y → X → α} {S : Set X}
    (hA : EquicontinuousOn (u ∘ (↑) : A → X → α) S) (hu : Continuous (S.restrict ∘ u)) :
    EquicontinuousOn (u ∘ (↑) : closure A → X → α) S :=
                                         /-
                                           X : Type u_3
                                           Y : Type u_5
                                           α : Type u_6
                                           tX : TopologicalSpace X
                                           tY : TopologicalSpace Y
                                           uα : UniformSpace α
                                           A : Set Y
                                           u : Y → X → α
                                           S : Set X
                                           hA : EquicontinuousOn (Function.comp u Subtype.val) S
                                           hu : Continuous (Function.comp S.restrict u)
                                           x : X
                                           hx : Membership.mem S x
                                           ⊢ Continuous (Function.comp (Function.eval x) u)
                                         -/
  fun x hx ↦ (hA x hx).closure' hu <| by exact continuous_apply ⟨x, hx⟩ |>.comp hu
                                         /-
                                           🎉 no goals
                                         -/


/-- If a set of functions is equicontinuous, its closure for the product topology is also
equicontinuous. -/
protected theorem Set.Equicontinuous.closure {A : Set <| X → α} (hA : A.Equicontinuous) :
    (closure A).Equicontinuous := fun x ↦ Set.EquicontinuousAt.closure (hA x)


/-- If a set of functions is equicontinuous, its closure for the product topology is also
equicontinuous. This would also be true for the coarser topology of pointwise convergence on `S`,
see `EquicontinuousOn.closure'`. -/
protected theorem Set.EquicontinuousOn.closure {A : Set <| X → α} {S : Set X}
    (hA : A.EquicontinuousOn S) : (closure A).EquicontinuousOn S :=
  fun x hx ↦ Set.EquicontinuousWithinAt.closure (hA x hx)


/-- If a set of functions is uniformly equicontinuous on a set `S`, the same is true for its
closure in *any* topology for which evaluation at any `x ∈ S` i continuous. Since this will be
applied to `DFunLike` types, we state it for any topological space with a map to `β → α` satisfying
the right continuity conditions. See also `Set.UniformEquicontinuousOn.closure` for a more familiar
(but weaker) statement. -/
theorem UniformEquicontinuousOn.closure' {A : Set Y} {u : Y → β → α} {S : Set β}
    (hA : UniformEquicontinuousOn (u ∘ (↑) : A → β → α) S) (hu : Continuous (S.restrict ∘ u)) :
    UniformEquicontinuousOn (u ∘ (↑) : closure A → β → α) S := by
  /-
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    ⊢ UniformEquicontinuousOn (Function.comp u Subtype.val) S
  -/
  intro U hU
  /-
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Filter.Eventually (fun xy => ∀ (i : ↑(closure A)), Membership.mem U { fst := …
  -/
  rcases mem_uniformity_isClosed hU with ⟨V, hV, hVclosed, hVU⟩
  /-
    case intro.intro.intro
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    ⊢ Filter.Eventually (fun xy => ∀ (i : ↑(closure A)), Membership.mem U { fst := …
  -/
  filter_upwards [hA V hV, mem_inf_of_right (mem_principal_self _)]
  /-
    case h
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    ⊢ ∀ (a : Prod β β), (∀ (i : ↑A), Membership.mem V { fst := Function.comp u Sub …
  -/
  rintro ⟨x, y⟩ hxy ⟨hxS, hyS⟩
  /-
    case h.mk.intro
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x y : β
    hxy : ∀ (i : ↑A), Membership.mem V { fst := Function.comp u Subtype.val i { fs …
    hxS : Membership.mem S { fst := x, snd := y }.1
    hyS : Membership.mem S { fst := x, snd := y }.2
    ⊢ ∀ (i : ↑(closure A)), Membership.mem U { fst := Function.comp u Subtype.val  …
  -/
  rw [SetCoe.forall] at *
  /-
    case h.mk.intro
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x y : β
    hxy : ∀ (x_1 : Y) (h : Membership.mem A x_1), Membership.mem V { fst := Functi …
    hxS : Membership.mem S { fst := x, snd := y }.1
    hyS : Membership.mem S { fst := x, snd := y }.2
    ⊢ ∀ (x_1 : Y) (h : Membership.mem (closure A) x_1), Membership.mem U { fst :=  …
  -/
  change A ⊆ (fun f => (u f x, u f y)) ⁻¹' V at hxy
  /-
    case h.mk.intro
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    S : Set β
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
    hu : Continuous (Function.comp S.restrict u)
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    x y : β
    hxS : Membership.mem S { fst := x, snd := y }.1
    hyS : Membership.mem S { fst := x, snd := y }.2
    hxy : HasSubset.Subset A (Set.preimage (fun f => { fst := u f x, snd := u f y  …
    ⊢ ∀ (x_1 : Y) (h : Membership.mem (closure A) x_1), Membership.mem U { fst :=  …
  -/
  refine (closure_minimal hxy <| hVclosed.preimage <| .prod_mk ?_ ?_).trans (preimage_mono hVU)
    /-
      case h.mk.intro.refine_1
      Y : Type u_5
      α : Type u_6
      β : Type u_8
      tY : TopologicalSpace Y
      uα : UniformSpace α
      uβ : UniformSpace β
      A : Set Y
      u : Y → β → α
      S : Set β
      hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
      hu : Continuous (Function.comp S.restrict u)
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      V : Set (Prod α α)
      hV : Membership.mem (uniformity α) V
      hVclosed : IsClosed V
      hVU : HasSubset.Subset V U
      x y : β
      hxS : Membership.mem S { fst := x, snd := y }.1
      hyS : Membership.mem S { fst := x, snd := y }.2
      hxy : HasSubset.Subset A (Set.preimage (fun f => { fst := u f x, snd := u f y  …
      ⊢ Continuous fun f => u f x
    -/
  · exact (continuous_apply ⟨x, hxS⟩).comp hu
    /-
      🎉 no goals
    -/
    /-
      case h.mk.intro.refine_2
      Y : Type u_5
      α : Type u_6
      β : Type u_8
      tY : TopologicalSpace Y
      uα : UniformSpace α
      uβ : UniformSpace β
      A : Set Y
      u : Y → β → α
      S : Set β
      hA : UniformEquicontinuousOn (Function.comp u Subtype.val) S
      hu : Continuous (Function.comp S.restrict u)
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      V : Set (Prod α α)
      hV : Membership.mem (uniformity α) V
      hVclosed : IsClosed V
      hVU : HasSubset.Subset V U
      x y : β
      hxS : Membership.mem S { fst := x, snd := y }.1
      hyS : Membership.mem S { fst := x, snd := y }.2
      hxy : HasSubset.Subset A (Set.preimage (fun f => { fst := u f x, snd := u f y  …
      ⊢ Continuous fun f => u f y
    -/
  · exact (continuous_apply ⟨y, hyS⟩).comp hu
    /-
      🎉 no goals
    -/


/-- If a set of functions is uniformly equicontinuous, the same is true for its closure in *any*
topology for which evaluation at any point is continuous. Since this will be applied to
`DFunLike` types, we state it for any topological space with a map to `β → α` satisfying the right
continuity conditions. See also `Set.UniformEquicontinuous.closure` for a more familiar statement.
-/
theorem UniformEquicontinuous.closure' {A : Set Y} {u : Y → β → α}
    (hA : UniformEquicontinuous (u ∘ (↑) : A → β → α)) (hu : Continuous u) :
    UniformEquicontinuous (u ∘ (↑) : closure A → β → α) := by
  /-
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    hA : UniformEquicontinuous (Function.comp u Subtype.val)
    hu : Continuous u
    ⊢ UniformEquicontinuous (Function.comp u Subtype.val)
  -/
  rw [← uniformEquicontinuousOn_univ] at hA ⊢
  /-
    Y : Type u_5
    α : Type u_6
    β : Type u_8
    tY : TopologicalSpace Y
    uα : UniformSpace α
    uβ : UniformSpace β
    A : Set Y
    u : Y → β → α
    hA : UniformEquicontinuousOn (Function.comp u Subtype.val) Set.univ
    hu : Continuous u
    ⊢ UniformEquicontinuousOn (Function.comp u Subtype.val) Set.univ
  -/
  exact hA.closure' (Pi.continuous_restrict _ |>.comp hu)
  /-
    🎉 no goals
  -/


/-- If a set of functions is uniformly equicontinuous, its closure for the product topology is also
uniformly equicontinuous. -/
protected theorem Set.UniformEquicontinuous.closure {A : Set <| β → α}
    (hA : A.UniformEquicontinuous) : (closure A).UniformEquicontinuous :=
  UniformEquicontinuous.closure' (u := id) hA continuous_id


/-- If a set of functions is uniformly equicontinuous on a set `S`, its closure for the product
topology is also uniformly equicontinuous. This would also be true for the coarser topology of
pointwise convergence on `S`, see `UniformEquicontinuousOn.closure'`. -/
protected theorem Set.UniformEquicontinuousOn.closure {A : Set <| β → α} {S : Set β}
    (hA : A.UniformEquicontinuousOn S) : (closure A).UniformEquicontinuousOn S :=
  UniformEquicontinuousOn.closure' (u := id) hA (Pi.continuous_restrict _)

/-
Implementation note: The following lemma (as well as all the following variations) could
theoretically be deduced from the "closure" statements above. For example, we could do:
```lean
theorem Filter.Tendsto.continuousAt_of_equicontinuousAt {l : Filter ι} [l.NeBot] {F : ι → X → α}
    {f : X → α} {x₀ : X} (h₁ : Tendsto F l (𝓝 f)) (h₂ : EquicontinuousAt F x₀) :
    ContinuousAt f x₀ :=
  (equicontinuousAt_iff_range.mp h₂).closure.continuousAt
    ⟨f, mem_closure_of_tendsto h₁ <| Eventually.of_forall mem_range_self⟩

theorem Filter.Tendsto.uniformContinuous_of_uniformEquicontinuous {l : Filter ι} [l.NeBot]
    {F : ι → β → α} {f : β → α} (h₁ : Tendsto F l (𝓝 f)) (h₂ : UniformEquicontinuous F) :
    UniformContinuous f :=
  (uniformEquicontinuous_iff_range.mp h₂).closure.uniformContinuous
    ⟨f, mem_closure_of_tendsto h₁ <| Eventually.of_forall mem_range_self⟩
```

Unfortunately, the proofs get painful when dealing with the relative case as one needs to change
the ambient topology. So it turns out to be easier to re-do the proof by hand.
-/


/-- If `𝓕 : ι → X → α` tends to `f : X → α` *pointwise on `S ∪ {x₀} : Set X`* along some nontrivial
filter, and if the family `𝓕` is equicontinuous at `x₀ : X` within `S`, then the limit is
continuous at `x₀` within `S`. -/
theorem Filter.Tendsto.continuousWithinAt_of_equicontinuousWithinAt {l : Filter ι} [l.NeBot]
    {F : ι → X → α} {f : X → α} {S : Set X} {x₀ : X} (h₁ : ∀ x ∈ S, Tendsto (F · x) l (𝓝 (f x)))
    (h₂ : Tendsto (F · x₀) l (𝓝 (f x₀))) (h₃ : EquicontinuousWithinAt F S x₀) :
    ContinuousWithinAt f S x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → X → α
    f : X → α
    S : Set X
    x₀ : X
    h₁ : ∀ (x : X), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : Filter.Tendsto (fun x => F x x₀) l (nhds (f x₀))
    h₃ : EquicontinuousWithinAt F S x₀
    ⊢ ContinuousWithinAt f S x₀
  -/
  intro U hU; rw [mem_map]
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → X → α
    f : X → α
    S : Set X
    x₀ : X
    h₁ : ∀ (x : X), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : Filter.Tendsto (fun x => F x x₀) l (nhds (f x₀))
    h₃ : EquicontinuousWithinAt F S x₀
    U : Set α
    hU : Membership.mem (nhds (f x₀)) U
    ⊢ Membership.mem (nhdsWithin x₀ S) (Set.preimage f U)
  -/
  rcases UniformSpace.mem_nhds_iff.mp hU with ⟨V, hV, hVU⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → X → α
    f : X → α
    S : Set X
    x₀ : X
    h₁ : ∀ (x : X), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : Filter.Tendsto (fun x => F x x₀) l (nhds (f x₀))
    h₃ : EquicontinuousWithinAt F S x₀
    U : Set α
    hU : Membership.mem (nhds (f x₀)) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVU : HasSubset.Subset (UniformSpace.ball (f x₀) V) U
    ⊢ Membership.mem (nhdsWithin x₀ S) (Set.preimage f U)
  -/
  rcases mem_uniformity_isClosed hV with ⟨W, hW, hWclosed, hWV⟩
  filter_upwards [h₃ W hW, eventually_mem_nhdsWithin] with x hx hxS using
    hVU <| ball_mono hWV (f x₀) <| hWclosed.mem_of_tendsto (h₂.prod_mk_nhds (h₁ x hxS)) <|
    Eventually.of_forall hx


/-- If `𝓕 : ι → X → α` tends to `f : X → α` *pointwise* along some nontrivial filter, and if the
family `𝓕` is equicontinuous at some `x₀ : X`, then the limit is continuous at `x₀`. -/
theorem Filter.Tendsto.continuousAt_of_equicontinuousAt {l : Filter ι} [l.NeBot] {F : ι → X → α}
    {f : X → α} {x₀ : X} (h₁ : Tendsto F l (𝓝 f)) (h₂ : EquicontinuousAt F x₀) :
    ContinuousAt f x₀ := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → X → α
    f : X → α
    x₀ : X
    h₁ : Filter.Tendsto F l (nhds f)
    h₂ : EquicontinuousAt F x₀
    ⊢ ContinuousAt f x₀
  -/
  rw [← continuousWithinAt_univ, ← equicontinuousWithinAt_univ, tendsto_pi_nhds] at *
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → X → α
    f : X → α
    x₀ : X
    h₁ : ∀ (x : X), Filter.Tendsto (fun i => F i x) l (nhds (f x))
    h₂ : EquicontinuousWithinAt F Set.univ x₀
    ⊢ ContinuousWithinAt f Set.univ x₀
  -/
  exact continuousWithinAt_of_equicontinuousWithinAt (fun x _ ↦ h₁ x) (h₁ x₀) h₂
  /-
    🎉 no goals
  -/


/-- If `𝓕 : ι → X → α` tends to `f : X → α` *pointwise* along some nontrivial filter, and if the
family `𝓕` is equicontinuous, then the limit is continuous. -/
theorem Filter.Tendsto.continuous_of_equicontinuous {l : Filter ι} [l.NeBot] {F : ι → X → α}
    {f : X → α} (h₁ : Tendsto F l (𝓝 f)) (h₂ : Equicontinuous F) : Continuous f :=
  continuous_iff_continuousAt.mpr fun x => h₁.continuousAt_of_equicontinuousAt (h₂ x)


/-- If `𝓕 : ι → X → α` tends to `f : X → α` *pointwise on `S : Set X`* along some nontrivial
filter, and if the family `𝓕` is equicontinuous, then the limit is continuous on `S`. -/
theorem Filter.Tendsto.continuousOn_of_equicontinuousOn {l : Filter ι} [l.NeBot] {F : ι → X → α}
    {f : X → α} {S : Set X} (h₁ : ∀ x ∈ S, Tendsto (F · x) l (𝓝 (f x)))
    (h₂ : EquicontinuousOn F S) : ContinuousOn f S :=
  fun x hx ↦ Filter.Tendsto.continuousWithinAt_of_equicontinuousWithinAt h₁ (h₁ x hx) (h₂ x hx)


/-- If `𝓕 : ι → β → α` tends to `f : β → α` *pointwise on `S : Set β`* along some nontrivial
filter, and if the family `𝓕` is uniformly equicontinuous on `S`, then the limit is uniformly
continuous on `S`. -/
theorem Filter.Tendsto.uniformContinuousOn_of_uniformEquicontinuousOn {l : Filter ι} [l.NeBot]
    {F : ι → β → α} {f : β → α} {S : Set β} (h₁ : ∀ x ∈ S, Tendsto (F · x) l (𝓝 (f x)))
    (h₂ : UniformEquicontinuousOn F S) :
    UniformContinuousOn f S := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → β → α
    f : β → α
    S : Set β
    h₁ : ∀ (x : β), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : UniformEquicontinuousOn F S
    ⊢ UniformContinuousOn f S
  -/
  intro U hU; rw [mem_map]
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → β → α
    f : β → α
    S : Set β
    h₁ : ∀ (x : β), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : UniformEquicontinuousOn F S
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Membership.mem (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))) …
  -/
  rcases mem_uniformity_isClosed hU with ⟨V, hV, hVclosed, hVU⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → β → α
    f : β → α
    S : Set β
    h₁ : ∀ (x : β), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : UniformEquicontinuousOn F S
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    ⊢ Membership.mem (Min.min (uniformity β) (Filter.principal (SProd.sprod S S))) …
  -/
  filter_upwards [h₂ V hV, mem_inf_of_right (mem_principal_self _)]
  /-
    case h
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → β → α
    f : β → α
    S : Set β
    h₁ : ∀ (x : β), Membership.mem S x → Filter.Tendsto (fun x_1 => F x_1 x) l (nh …
    h₂ : UniformEquicontinuousOn F S
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVclosed : IsClosed V
    hVU : HasSubset.Subset V U
    ⊢ ∀ (a : Prod β β), (∀ (i : ι), Membership.mem V { fst := F i a.1, snd := F i  …
  -/
  rintro ⟨x, y⟩ hxy ⟨hxS, hyS⟩
  exact hVU <| hVclosed.mem_of_tendsto ((h₁ x hxS).prod_mk_nhds (h₁ y hyS)) <|
    Eventually.of_forall hxy


/-- If `𝓕 : ι → β → α` tends to `f : β → α` *pointwise* along some nontrivial filter, and if the
family `𝓕` is uniformly equicontinuous, then the limit is uniformly continuous. -/
theorem Filter.Tendsto.uniformContinuous_of_uniformEquicontinuous {l : Filter ι} [l.NeBot]
    {F : ι → β → α} {f : β → α} (h₁ : Tendsto F l (𝓝 f)) (h₂ : UniformEquicontinuous F) :
    UniformContinuous f := by
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → β → α
    f : β → α
    h₁ : Filter.Tendsto F l (nhds f)
    h₂ : UniformEquicontinuous F
    ⊢ UniformContinuous f
  -/
  rw [← uniformContinuousOn_univ, ← uniformEquicontinuousOn_univ, tendsto_pi_nhds] at *
  /-
    ι : Type u_1
    α : Type u_6
    β : Type u_8
    uα : UniformSpace α
    uβ : UniformSpace β
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → β → α
    f : β → α
    h₁ : ∀ (x : β), Filter.Tendsto (fun i => F i x) l (nhds (f x))
    h₂ : UniformEquicontinuousOn F Set.univ
    ⊢ UniformContinuousOn f Set.univ
  -/
  exact uniformContinuousOn_of_uniformEquicontinuousOn (fun x _ ↦ h₁ x) h₂
  /-
    🎉 no goals
  -/


/-- If `F : ι → X → α` is a family of functions equicontinuous at `x`,
it tends to `f y` along a filter `l` for any `y ∈ s`,
the limit function `f` tends to `z` along `𝓝[s] x`, and `x ∈ closure s`,
then `(F · x)` tends to `z` along `l`.

In some sense, this is a converse of `EquicontinuousAt.closure`. -/
theorem EquicontinuousAt.tendsto_of_mem_closure {l : Filter ι} {F : ι → X → α} {f : X → α}
    {s : Set X} {x : X} {z : α} (hF : EquicontinuousAt F x) (hf : Tendsto f (𝓝[s] x) (𝓝 z))
    (hs : ∀ y ∈ s, Tendsto (F · y) l (𝓝 (f y))) (hx : x ∈ closure s) :
    Tendsto (F · x) l (𝓝 z) := by
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : Filter.Tendsto f (nhdsWithin x s) (nhds z)
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : Membership.mem (closure s) x
    ⊢ Filter.Tendsto (fun x_1 => F x_1 x) l (nhds z)
  -/
  rw [(nhds_basis_uniformity (𝓤 α).basis_sets).tendsto_right_iff] at hf ⊢
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventual …
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : Membership.mem (closure s) x
    ⊢ ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventually  …
  -/
  intro U hU
  /-
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventual …
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : Membership.mem (closure s) x
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (setOf fun y => Membership.mem  …
  -/
  rcases comp_comp_symm_mem_uniformity_sets hU with ⟨V, hV, hVs, hVU⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventual …
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : Membership.mem (closure s) x
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVs : SymmetricRel V
    hVU : HasSubset.Subset (compRel (compRel V V) V) U
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (setOf fun y => Membership.mem  …
  -/
  rw [mem_closure_iff_nhdsWithin_neBot] at hx
  have : ∀ᶠ y in 𝓝[s] x, y ∈ s ∧ (∀ i, (F i x, F i y) ∈ V) ∧ (f y, z) ∈ V :=
    eventually_mem_nhdsWithin.and <| ((hF V hV).filter_mono nhdsWithin_le_nhds).and (hf V hV)
  /-
    case intro.intro.intro
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventual …
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : (nhdsWithin x s).NeBot
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVs : SymmetricRel V
    hVU : HasSubset.Subset (compRel (compRel V V) V) U
    this : Filter.Eventually (fun y => And (Membership.mem s y) (And (∀ (i : ι), M …
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (setOf fun y => Membership.mem  …
  -/
  rcases this.exists with ⟨y, hys, hFy, hfy⟩
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventual …
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : (nhdsWithin x s).NeBot
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVs : SymmetricRel V
    hVU : HasSubset.Subset (compRel (compRel V V) V) U
    this : Filter.Eventually (fun y => And (Membership.mem s y) (And (∀ (i : ι), M …
    y : X
    hys : Membership.mem s y
    hFy : ∀ (i : ι), Membership.mem V { fst := F i x, snd := F i y }
    hfy : Membership.mem V { fst := f y, snd := z }
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (setOf fun y => Membership.mem  …
  -/
  filter_upwards [hs y hys (ball_mem_nhds _ hV)] with i hi
  /-
    case h
    ι : Type u_1
    X : Type u_3
    α : Type u_6
    tX : TopologicalSpace X
    uα : UniformSpace α
    l : Filter ι
    F : ι → X → α
    f : X → α
    s : Set X
    x : X
    z : α
    hF : EquicontinuousAt F x
    hf : ∀ (i : Set (Prod α α)), Membership.mem (uniformity α) i → Filter.Eventual …
    hs : ∀ (y : X), Membership.mem s y → Filter.Tendsto (fun x => F x y) l (nhds ( …
    hx : (nhdsWithin x s).NeBot
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    V : Set (Prod α α)
    hV : Membership.mem (uniformity α) V
    hVs : SymmetricRel V
    hVU : HasSubset.Subset (compRel (compRel V V) V) U
    this : Filter.Eventually (fun y => And (Membership.mem s y) (And (∀ (i : ι), M …
    y : X
    hys : Membership.mem s y
    hFy : ∀ (i : ι), Membership.mem V { fst := F i x, snd := F i y }
    hfy : Membership.mem V { fst := f y, snd := z }
    i : ι
    hi : Membership.mem (Set.preimage (fun x => F x y) (UniformSpace.ball (f y) V) …
    ⊢ Membership.mem (id U) { fst := F i x, snd := z }
  -/
  exact hVU ⟨_, ⟨_, hFy i, (mem_ball_symmetry hVs).2 hi⟩, hfy⟩
  /-
    🎉 no goals
  -/


/-- If `F : ι → X → α` is an equicontinuous family of functions,
`f : X → α` is a continuous function, and `l` is a filter on `ι`,
then `{x | Filter.Tendsto (F · x) l (𝓝 (f x))}` is a closed set. -/
theorem Equicontinuous.isClosed_setOf_tendsto {l : Filter ι} {F : ι → X → α} {f : X → α}
    (hF : Equicontinuous F) (hf : Continuous f) :
    IsClosed {x | Tendsto (F · x) l (𝓝 (f x))} :=
  closure_subset_iff_isClosed.mp fun x hx ↦
    (hF x).tendsto_of_mem_closure (hf.continuousAt.mono_left inf_le_left) (fun _ ↦ id) hx


