/-- Class `ContinuousConstSMul Γ T` says that the scalar multiplication `(•) : Γ → T → T`
is continuous in the second argument. We use the same class for all kinds of multiplicative
actions, including (semi)modules and algebras.

Note that both `ContinuousConstSMul α α` and `ContinuousConstSMul αᵐᵒᵖ α` are
weaker versions of `ContinuousMul α`. -/
class ContinuousConstSMul (Γ : Type*) (T : Type*) [TopologicalSpace T] [SMul Γ T] : Prop where
  /-- The scalar multiplication `(•) : Γ → T → T` is continuous in the second argument. -/
  continuous_const_smul : ∀ γ : Γ, Continuous fun x : T => γ • x


/-- Class `ContinuousConstVAdd Γ T` says that the additive action `(+ᵥ) : Γ → T → T`
is continuous in the second argument. We use the same class for all kinds of additive actions,
including (semi)modules and algebras.

Note that both `ContinuousConstVAdd α α` and `ContinuousConstVAdd αᵐᵒᵖ α` are
weaker versions of `ContinuousVAdd α`. -/
class ContinuousConstVAdd (Γ : Type*) (T : Type*) [TopologicalSpace T] [VAdd Γ T] : Prop where
  /-- The additive action `(+ᵥ) : Γ → T → T` is continuous in the second argument. -/
  continuous_const_vadd : ∀ γ : Γ, Continuous fun x : T => γ +ᵥ x


@[to_additive]
instance : ContinuousConstSMul (ULift M) α := ⟨fun γ ↦ continuous_const_smul (ULift.down γ)⟩


@[to_additive]
theorem Filter.Tendsto.const_smul {f : β → α} {l : Filter β} {a : α} (hf : Tendsto f l (𝓝 a))
    (c : M) : Tendsto (fun x => c • f x) l (𝓝 (c • a)) :=
  ((continuous_const_smul _).tendsto _).comp hf


@[to_additive]
nonrec theorem ContinuousWithinAt.const_smul (hg : ContinuousWithinAt g s b) (c : M) :
    ContinuousWithinAt (fun x => c • g x) s b :=
  hg.const_smul c


@[to_additive (attr := fun_prop)]
nonrec theorem ContinuousAt.const_smul (hg : ContinuousAt g b) (c : M) :
    ContinuousAt (fun x => c • g x) b :=
  hg.const_smul c


@[to_additive (attr := fun_prop)]
theorem ContinuousOn.const_smul (hg : ContinuousOn g s) (c : M) :
    ContinuousOn (fun x => c • g x) s := fun x hx => (hg x hx).const_smul c


@[to_additive (attr := continuity, fun_prop)]
theorem Continuous.const_smul (hg : Continuous g) (c : M) : Continuous fun x => c • g x :=
  (continuous_const_smul _).comp hg


/-- If a scalar is central, then its right action is continuous when its left action is. -/
@[to_additive "If an additive action is central, then its right action is continuous when its left
action is."]
instance ContinuousConstSMul.op [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] :
    ContinuousConstSMul Mᵐᵒᵖ α :=
                                /-
                                  M : Type u_1
                                  α : Type u_2
                                  β : Type u_3
                                  inst✝⁵ : TopologicalSpace α
                                  inst✝⁴ : SMul M α
                                  inst✝³ : ContinuousConstSMul M α
                                  inst✝² : TopologicalSpace β
                                  g : β → α
                                  b : β
                                  s : Set β
                                  inst✝¹ : SMul (MulOpposite M) α
                                  inst✝ : IsCentralScalar M α
                                  c : M
                                  ⊢ Continuous fun x => HSMul.hSMul (MulOpposite.op c) x
                                -/
  ⟨MulOpposite.rec' fun c => by simpa only [op_smul_eq_smul] using continuous_const_smul c⟩
                                /-
                                  🎉 no goals
                                -/


@[to_additive]
instance MulOpposite.continuousConstSMul : ContinuousConstSMul M αᵐᵒᵖ :=
  ⟨fun c => MulOpposite.continuous_op.comp <| MulOpposite.continuous_unop.const_smul c⟩


@[to_additive]
instance : ContinuousConstSMul M αᵒᵈ := ‹ContinuousConstSMul M α›


@[to_additive]
instance OrderDual.continuousConstSMul' : ContinuousConstSMul Mᵒᵈ α :=
  ‹ContinuousConstSMul M α›


@[to_additive]
instance Prod.continuousConstSMul [SMul M β] [ContinuousConstSMul M β] :
    ContinuousConstSMul M (α × β) :=
  ⟨fun _ => (continuous_fst.const_smul _).prod_mk (continuous_snd.const_smul _)⟩


@[to_additive]
instance {ι : Type*} {γ : ι → Type*} [∀ i, TopologicalSpace (γ i)] [∀ i, SMul M (γ i)]
    [∀ i, ContinuousConstSMul M (γ i)] : ContinuousConstSMul M (∀ i, γ i) :=
  ⟨fun _ => continuous_pi fun i => (continuous_apply i).const_smul _⟩


@[to_additive]
theorem IsCompact.smul {α β} [SMul α β] [TopologicalSpace β] [ContinuousConstSMul α β] (a : α)
    {s : Set β} (hs : IsCompact s) : IsCompact (a • s) :=
  hs.image (continuous_id.const_smul a)


@[to_additive]
theorem Specializes.const_smul {x y : α} (h : x ⤳ y) (c : M) : (c • x) ⤳ (c • y) :=
  h.map (continuous_const_smul c)


@[to_additive]
theorem Inseparable.const_smul {x y : α} (h : Inseparable x y) (c : M) :
    Inseparable (c • x) (c • y) :=
  h.map (continuous_const_smul c)


@[to_additive]
theorem Topology.IsInducing.continuousConstSMul {N β : Type*} [SMul N β] [TopologicalSpace β]
    {g : β → α} (hg : IsInducing g) (f : N → M) (hf : ∀ {c : N} {x : β}, g (c • x) = f c • g x) :
    ContinuousConstSMul N β where
  continuous_const_smul c := by
    /-
      M : Type u_1
      α : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : SMul M α
      inst✝² : ContinuousConstSMul M α
      N : Type u_4
      β : Type u_5
      inst✝¹ : SMul N β
      inst✝ : TopologicalSpace β
      g : β → α
      hg : Topology.IsInducing g
      f : N → M
      hf : ∀ {c : N} {x : β}, Eq (g (HSMul.hSMul c x)) (HSMul.hSMul (f c) (g x))
      c : N
      ⊢ Continuous fun x => HSMul.hSMul c x
    -/
    simpa only [Function.comp_def, hf, hg.continuous_iff] using hg.continuous.const_smul (f c)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")]
alias Inducing.continuousConstSMul := IsInducing.continuousConstSMul


@[to_additive]
instance Units.continuousConstSMul : ContinuousConstSMul Mˣ α where
  continuous_const_smul m := (continuous_const_smul (m : M) : _)


@[to_additive]
theorem smul_closure_subset (c : M) (s : Set α) : c • closure s ⊆ closure (c • s) :=
  ((Set.mapsTo_image _ _).closure <| continuous_const_smul c).image_subset


@[to_additive]
theorem smul_closure_orbit_subset (c : M) (x : α) :
    c • closure (MulAction.orbit M x) ⊆ closure (MulAction.orbit M x) :=
  (smul_closure_subset c _).trans <| closure_mono <| MulAction.smul_orbit_subset _ _


theorem isClosed_setOf_map_smul {N : Type*} [Monoid N] (α β) [MulAction M α] [MulAction N β]
    [TopologicalSpace β] [T2Space β] [ContinuousConstSMul N β] (σ : M → N) :
    IsClosed { f : α → β | ∀ c x, f (c • x) = σ c • f x } := by
  /-
    M : Type u_1
    inst✝⁶ : Monoid M
    N : Type u_4
    inst✝⁵ : Monoid N
    α : Type u_5
    β : Type u_6
    inst✝⁴ : MulAction M α
    inst✝³ : MulAction N β
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : ContinuousConstSMul N β
    σ : M → N
    ⊢ IsClosed (setOf fun f => ∀ (c : M) (x : α), Eq (f (HSMul.hSMul c x)) (HSMul. …
  -/
  simp only [Set.setOf_forall]
  exact isClosed_iInter fun c => isClosed_iInter fun x =>
    isClosed_eq (continuous_apply _) ((continuous_apply _).const_smul _)


@[to_additive]
theorem tendsto_const_smul_iff {f : β → α} {l : Filter β} {a : α} (c : G) :
    Tendsto (fun x => c • f x) l (𝓝 <| c • a) ↔ Tendsto f l (𝓝 a) :=
               /-
                 α : Type u_2
                 β : Type u_3
                 G : Type u_4
                 inst✝³ : TopologicalSpace α
                 inst✝² : Group G
                 inst✝¹ : MulAction G α
                 inst✝ : ContinuousConstSMul G α
                 f : β → α
                 l : Filter β
                 a : α
                 c : G
                 h : Filter.Tendsto (fun x => HSMul.hSMul c (f x)) l (nhds (HSMul.hSMul c a))
                 ⊢ Filter.Tendsto f l (nhds a)
               -/
  ⟨fun h => by simpa only [inv_smul_smul] using h.const_smul c⁻¹, fun h => h.const_smul _⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
theorem continuousWithinAt_const_smul_iff (c : G) :
    ContinuousWithinAt (fun x => c • f x) s b ↔ ContinuousWithinAt f s b :=
  tendsto_const_smul_iff c


@[to_additive]
theorem continuousOn_const_smul_iff (c : G) :
    ContinuousOn (fun x => c • f x) s ↔ ContinuousOn f s :=
  forall₂_congr fun _ _ => continuousWithinAt_const_smul_iff c


@[to_additive]
theorem continuousAt_const_smul_iff (c : G) :
    ContinuousAt (fun x => c • f x) b ↔ ContinuousAt f b :=
  tendsto_const_smul_iff c


@[to_additive]
theorem continuous_const_smul_iff (c : G) : (Continuous fun x => c • f x) ↔ Continuous f := by
  /-
    α : Type u_2
    β : Type u_3
    G : Type u_4
    inst✝⁴ : TopologicalSpace α
    inst✝³ : Group G
    inst✝² : MulAction G α
    inst✝¹ : ContinuousConstSMul G α
    inst✝ : TopologicalSpace β
    f : β → α
    c : G
    ⊢ Iff (Continuous fun x => HSMul.hSMul c (f x)) (Continuous f)
  -/
  simp only [continuous_iff_continuousAt, continuousAt_const_smul_iff]
  /-
    🎉 no goals
  -/


/-- The homeomorphism given by scalar multiplication by a given element of a group `Γ` acting on
  `T` is a homeomorphism from `T` to itself. -/
@[to_additive (attr := simps!)]
def Homeomorph.smul (γ : G) : α ≃ₜ α where
  toEquiv := MulAction.toPerm γ
  continuous_toFun := continuous_const_smul γ
  continuous_invFun := continuous_const_smul γ⁻¹


@[to_additive]
theorem isOpenMap_smul (c : G) : IsOpenMap fun x : α => c • x :=
  (Homeomorph.smul c).isOpenMap


@[to_additive]
theorem IsOpen.smul {s : Set α} (hs : IsOpen s) (c : G) : IsOpen (c • s) :=
  isOpenMap_smul c s hs


@[to_additive]
theorem isClosedMap_smul (c : G) : IsClosedMap fun x : α => c • x :=
  (Homeomorph.smul c).isClosedMap


@[to_additive]
theorem IsClosed.smul {s : Set α} (hs : IsClosed s) (c : G) : IsClosed (c • s) :=
  isClosedMap_smul c s hs


@[to_additive]
theorem closure_smul (c : G) (s : Set α) : closure (c • s) = c • closure s :=
  ((Homeomorph.smul c).image_closure s).symm


@[to_additive]
theorem Dense.smul (c : G) {s : Set α} (hs : Dense s) : Dense (c • s) := by
  /-
    α : Type u_2
    G : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : ContinuousConstSMul G α
    c : G
    s : Set α
    hs : Dense s
    ⊢ Dense (HSMul.hSMul c s)
  -/
  rw [dense_iff_closure_eq] at hs ⊢; rw [closure_smul, hs, smul_set_univ]
                                     /-
                                       🎉 no goals
                                     -/


@[to_additive]
theorem interior_smul (c : G) (s : Set α) : interior (c • s) = c • interior s :=
  ((Homeomorph.smul c).image_interior s).symm


@[to_additive]
theorem IsOpen.smul_left {s : Set G} {t : Set α} (ht : IsOpen t) : IsOpen (s • t) := by
  /-
    α : Type u_2
    G : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : ContinuousConstSMul G α
    s : Set G
    t : Set α
    ht : IsOpen t
    ⊢ IsOpen (HSMul.hSMul s t)
  -/
  rw [← iUnion_smul_set]
  /-
    α : Type u_2
    G : Type u_4
    inst✝³ : TopologicalSpace α
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : ContinuousConstSMul G α
    s : Set G
    t : Set α
    ht : IsOpen t
    ⊢ IsOpen (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a t)
  -/
  exact isOpen_biUnion fun a _ => ht.smul _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem subset_interior_smul_right {s : Set G} {t : Set α} : s • interior t ⊆ interior (s • t) :=
  interior_maximal (Set.smul_subset_smul_left interior_subset) isOpen_interior.smul_left


@[to_additive (attr := simp)]
theorem smul_mem_nhds_smul_iff {t : Set α} (g : G) {a : α} : g • t ∈ 𝓝 (g • a) ↔ t ∈ 𝓝 a :=
  (Homeomorph.smul g).isOpenEmbedding.image_mem_nhds


@[to_additive] alias ⟨_, smul_mem_nhds_smul⟩ := smul_mem_nhds_smul_iff


@[to_additive (attr := deprecated "No deprecation message was provided." (since := "2024-08-06"))]
alias smul_mem_nhds := smul_mem_nhds_smul


@[to_additive (attr := simp)]
theorem smul_mem_nhds_self [TopologicalSpace G] [ContinuousConstSMul G G] {g : G} {s : Set G} :
    g • s ∈ 𝓝 g ↔ s ∈ 𝓝 1 := by
  /-
    G : Type u_4
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousConstSMul G G
    g : G
    s : Set G
    ⊢ Iff (Membership.mem (nhds g) (HSMul.hSMul g s)) (Membership.mem (nhds 1) s)
  -/
  rw [← smul_mem_nhds_smul_iff g⁻¹]; simp
                                     /-
                                       🎉 no goals
                                     -/


theorem tendsto_const_smul_iff₀ {f : β → α} {l : Filter β} {a : α} {c : G₀} (hc : c ≠ 0) :
    Tendsto (fun x => c • f x) l (𝓝 <| c • a) ↔ Tendsto f l (𝓝 a) :=
  tendsto_const_smul_iff (Units.mk0 c hc)


theorem continuousWithinAt_const_smul_iff₀ (hc : c ≠ 0) :
    ContinuousWithinAt (fun x => c • f x) s b ↔ ContinuousWithinAt f s b :=
  tendsto_const_smul_iff (Units.mk0 c hc)


theorem continuousOn_const_smul_iff₀ (hc : c ≠ 0) :
    ContinuousOn (fun x => c • f x) s ↔ ContinuousOn f s :=
  continuousOn_const_smul_iff (Units.mk0 c hc)


theorem continuousAt_const_smul_iff₀ (hc : c ≠ 0) :
    ContinuousAt (fun x => c • f x) b ↔ ContinuousAt f b :=
  continuousAt_const_smul_iff (Units.mk0 c hc)


theorem continuous_const_smul_iff₀ (hc : c ≠ 0) : (Continuous fun x => c • f x) ↔ Continuous f :=
  continuous_const_smul_iff (Units.mk0 c hc)


/-- Scalar multiplication by a non-zero element of a group with zero acting on `α` is a
homeomorphism from `α` onto itself. -/
@[simps! (config := .asFn) apply]
protected def Homeomorph.smulOfNeZero (c : G₀) (hc : c ≠ 0) : α ≃ₜ α :=
  Homeomorph.smul (Units.mk0 c hc)


@[simp]
theorem Homeomorph.smulOfNeZero_symm_apply {c : G₀} (hc : c ≠ 0) :
    ⇑(Homeomorph.smulOfNeZero c hc).symm = (c⁻¹ • · : α → α) :=
  rfl


theorem isOpenMap_smul₀ {c : G₀} (hc : c ≠ 0) : IsOpenMap fun x : α => c • x :=
  (Homeomorph.smulOfNeZero c hc).isOpenMap


theorem IsOpen.smul₀ {c : G₀} {s : Set α} (hs : IsOpen s) (hc : c ≠ 0) : IsOpen (c • s) :=
  isOpenMap_smul₀ hc s hs


theorem interior_smul₀ {c : G₀} (hc : c ≠ 0) (s : Set α) : interior (c • s) = c • interior s :=
  ((Homeomorph.smulOfNeZero c hc).image_interior s).symm


theorem closure_smul₀' {c : G₀} (hc : c ≠ 0) (s : Set α) :
    closure (c • s) = c • closure s :=
  ((Homeomorph.smulOfNeZero c hc).image_closure s).symm


theorem closure_smul₀ {E} [Zero E] [MulActionWithZero G₀ E] [TopologicalSpace E] [T1Space E]
    [ContinuousConstSMul G₀ E] (c : G₀) (s : Set E) : closure (c • s) = c • closure s := by
  /-
    G₀ : Type u_4
    inst✝⁵ : GroupWithZero G₀
    E : Type u_5
    inst✝⁴ : Zero E
    inst✝³ : MulActionWithZero G₀ E
    inst✝² : TopologicalSpace E
    inst✝¹ : T1Space E
    inst✝ : ContinuousConstSMul G₀ E
    c : G₀
    s : Set E
    ⊢ Eq (closure (HSMul.hSMul c s)) (HSMul.hSMul c (closure s))
  -/
  rcases eq_or_ne c 0 with (rfl | hc)
    /-
      case inl
      G₀ : Type u_4
      inst✝⁵ : GroupWithZero G₀
      E : Type u_5
      inst✝⁴ : Zero E
      inst✝³ : MulActionWithZero G₀ E
      inst✝² : TopologicalSpace E
      inst✝¹ : T1Space E
      inst✝ : ContinuousConstSMul G₀ E
      s : Set E
      ⊢ Eq (closure (HSMul.hSMul 0 s)) (HSMul.hSMul 0 (closure s))
    -/
  · rcases eq_empty_or_nonempty s with (rfl | hs)
      /-
        case inl.inl
        G₀ : Type u_4
        inst✝⁵ : GroupWithZero G₀
        E : Type u_5
        inst✝⁴ : Zero E
        inst✝³ : MulActionWithZero G₀ E
        inst✝² : TopologicalSpace E
        inst✝¹ : T1Space E
        inst✝ : ContinuousConstSMul G₀ E
        ⊢ Eq (closure (HSMul.hSMul 0 EmptyCollection.emptyCollection)) (HSMul.hSMul 0  …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        G₀ : Type u_4
        inst✝⁵ : GroupWithZero G₀
        E : Type u_5
        inst✝⁴ : Zero E
        inst✝³ : MulActionWithZero G₀ E
        inst✝² : TopologicalSpace E
        inst✝¹ : T1Space E
        inst✝ : ContinuousConstSMul G₀ E
        s : Set E
        hs : s.Nonempty
        ⊢ Eq (closure (HSMul.hSMul 0 s)) (HSMul.hSMul 0 (closure s))
      -/
    · rw [zero_smul_set hs, zero_smul_set hs.closure]
      /-
        case inl.inr
        G₀ : Type u_4
        inst✝⁵ : GroupWithZero G₀
        E : Type u_5
        inst✝⁴ : Zero E
        inst✝³ : MulActionWithZero G₀ E
        inst✝² : TopologicalSpace E
        inst✝¹ : T1Space E
        inst✝ : ContinuousConstSMul G₀ E
        s : Set E
        hs : s.Nonempty
        ⊢ Eq (closure 0) 0
      -/
      exact closure_singleton
      /-
        🎉 no goals
      -/
    /-
      case inr
      G₀ : Type u_4
      inst✝⁵ : GroupWithZero G₀
      E : Type u_5
      inst✝⁴ : Zero E
      inst✝³ : MulActionWithZero G₀ E
      inst✝² : TopologicalSpace E
      inst✝¹ : T1Space E
      inst✝ : ContinuousConstSMul G₀ E
      c : G₀
      s : Set E
      hc : Ne c 0
      ⊢ Eq (closure (HSMul.hSMul c s)) (HSMul.hSMul c (closure s))
    -/
  · exact closure_smul₀' hc s
    /-
      🎉 no goals
    -/


/-- `smul` is a closed map in the second argument.

The lemma that `smul` is a closed map in the first argument (for a normed space over a complete
normed field) is `isClosedMap_smul_left` in `Analysis.Normed.Module.FiniteDimension`. -/
theorem isClosedMap_smul_of_ne_zero {c : G₀} (hc : c ≠ 0) : IsClosedMap fun x : α => c • x :=
  (Homeomorph.smulOfNeZero c hc).isClosedMap


theorem IsClosed.smul_of_ne_zero {c : G₀} {s : Set α} (hs : IsClosed s) (hc : c ≠ 0) :
    IsClosed (c • s) :=
  isClosedMap_smul_of_ne_zero hc s hs


/-- `smul` is a closed map in the second argument.

The lemma that `smul` is a closed map in the first argument (for a normed space over a complete
normed field) is `isClosedMap_smul_left` in `Analysis.Normed.Module.FiniteDimension`. -/
theorem isClosedMap_smul₀ {E : Type*} [Zero E] [MulActionWithZero G₀ E] [TopologicalSpace E]
    [T1Space E] [ContinuousConstSMul G₀ E] (c : G₀) : IsClosedMap fun x : E => c • x := by
  /-
    G₀ : Type u_4
    inst✝⁵ : GroupWithZero G₀
    E : Type u_5
    inst✝⁴ : Zero E
    inst✝³ : MulActionWithZero G₀ E
    inst✝² : TopologicalSpace E
    inst✝¹ : T1Space E
    inst✝ : ContinuousConstSMul G₀ E
    c : G₀
    ⊢ IsClosedMap fun x => HSMul.hSMul c x
  -/
  rcases eq_or_ne c 0 with (rfl | hne)
    /-
      case inl
      G₀ : Type u_4
      inst✝⁵ : GroupWithZero G₀
      E : Type u_5
      inst✝⁴ : Zero E
      inst✝³ : MulActionWithZero G₀ E
      inst✝² : TopologicalSpace E
      inst✝¹ : T1Space E
      inst✝ : ContinuousConstSMul G₀ E
      ⊢ IsClosedMap fun x => HSMul.hSMul 0 x
    -/
  · simp only [zero_smul]
    /-
      case inl
      G₀ : Type u_4
      inst✝⁵ : GroupWithZero G₀
      E : Type u_5
      inst✝⁴ : Zero E
      inst✝³ : MulActionWithZero G₀ E
      inst✝² : TopologicalSpace E
      inst✝¹ : T1Space E
      inst✝ : ContinuousConstSMul G₀ E
      ⊢ IsClosedMap fun x => 0
    -/
    exact isClosedMap_const
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_4
      inst✝⁵ : GroupWithZero G₀
      E : Type u_5
      inst✝⁴ : Zero E
      inst✝³ : MulActionWithZero G₀ E
      inst✝² : TopologicalSpace E
      inst✝¹ : T1Space E
      inst✝ : ContinuousConstSMul G₀ E
      c : G₀
      hne : Ne c 0
      ⊢ IsClosedMap fun x => HSMul.hSMul c x
    -/
  · exact (Homeomorph.smulOfNeZero c hne).isClosedMap
    /-
      🎉 no goals
    -/


theorem IsClosed.smul₀ {E : Type*} [Zero E] [MulActionWithZero G₀ E] [TopologicalSpace E]
    [T1Space E] [ContinuousConstSMul G₀ E] (c : G₀) {s : Set E} (hs : IsClosed s) :
    IsClosed (c • s) :=
  isClosedMap_smul₀ c s hs


theorem HasCompactMulSupport.comp_smul {β : Type*} [One β] {f : α → β} (h : HasCompactMulSupport f)
    {c : G₀} (hc : c ≠ 0) : HasCompactMulSupport fun x => f (c • x) :=
  h.comp_homeomorph (Homeomorph.smulOfNeZero c hc)


theorem HasCompactSupport.comp_smul {β : Type*} [Zero β] {f : α → β} (h : HasCompactSupport f)
    {c : G₀} (hc : c ≠ 0) : HasCompactSupport fun x => f (c • x) :=
  h.comp_homeomorph (Homeomorph.smulOfNeZero c hc)


nonrec theorem tendsto_const_smul_iff {f : β → α} {l : Filter β} {a : α} {c : M} (hc : IsUnit c) :
    Tendsto (fun x => c • f x) l (𝓝 <| c • a) ↔ Tendsto f l (𝓝 a) :=
  tendsto_const_smul_iff hc.unit


nonrec theorem continuousWithinAt_const_smul_iff (hc : IsUnit c) :
    ContinuousWithinAt (fun x => c • f x) s b ↔ ContinuousWithinAt f s b :=
  continuousWithinAt_const_smul_iff hc.unit


nonrec theorem continuousOn_const_smul_iff (hc : IsUnit c) :
    ContinuousOn (fun x => c • f x) s ↔ ContinuousOn f s :=
  continuousOn_const_smul_iff hc.unit


nonrec theorem continuousAt_const_smul_iff (hc : IsUnit c) :
    ContinuousAt (fun x => c • f x) b ↔ ContinuousAt f b :=
  continuousAt_const_smul_iff hc.unit


nonrec theorem continuous_const_smul_iff (hc : IsUnit c) :
    (Continuous fun x => c • f x) ↔ Continuous f :=
  continuous_const_smul_iff hc.unit


nonrec theorem isOpenMap_smul (hc : IsUnit c) : IsOpenMap fun x : α => c • x :=
  isOpenMap_smul hc.unit


nonrec theorem isClosedMap_smul (hc : IsUnit c) : IsClosedMap fun x : α => c • x :=
  isClosedMap_smul hc.unit


nonrec theorem smul_mem_nhds_smul_iff (hc : IsUnit c) {s : Set α} {a : α} :
    c • s ∈ 𝓝 (c • a) ↔ s ∈ 𝓝 a :=
  smul_mem_nhds_smul_iff hc.unit


/-- Class `ProperlyDiscontinuousSMul Γ T` says that the scalar multiplication `(•) : Γ → T → T`
is properly discontinuous, that is, for any pair of compact sets `K, L` in `T`, only finitely many
`γ:Γ` move `K` to have nontrivial intersection with `L`.
-/
class ProperlyDiscontinuousSMul (Γ : Type*) (T : Type*) [TopologicalSpace T] [SMul Γ T] :
    Prop where
  /-- Given two compact sets `K` and `L`, `γ • K ∩ L` is nonempty for finitely many `γ`. -/
  finite_disjoint_inter_image :
    ∀ {K L : Set T}, IsCompact K → IsCompact L → Set.Finite { γ : Γ | (γ • ·) '' K ∩ L ≠ ∅ }


/-- Class `ProperlyDiscontinuousVAdd Γ T` says that the additive action `(+ᵥ) : Γ → T → T`
is properly discontinuous, that is, for any pair of compact sets `K, L` in `T`, only finitely many
`γ:Γ` move `K` to have nontrivial intersection with `L`.
-/
class ProperlyDiscontinuousVAdd (Γ : Type*) (T : Type*) [TopologicalSpace T] [VAdd Γ T] :
  Prop where
  /-- Given two compact sets `K` and `L`, `γ +ᵥ K ∩ L` is nonempty for finitely many `γ`. -/
  finite_disjoint_inter_image :
    ∀ {K L : Set T}, IsCompact K → IsCompact L → Set.Finite { γ : Γ | (γ +ᵥ ·) '' K ∩ L ≠ ∅ }


/-- A finite group action is always properly discontinuous. -/
@[to_additive "A finite group action is always properly discontinuous."]
instance (priority := 100) Finite.to_properlyDiscontinuousSMul [Finite Γ] :
    ProperlyDiscontinuousSMul Γ T where finite_disjoint_inter_image _ _ := Set.toFinite _


/-- The quotient map by a group action is open, i.e. the quotient by a group action is an open
  quotient. -/
@[to_additive "The quotient map by a group action is open, i.e. the quotient by a group
action is an open quotient. "]
theorem isOpenMap_quotient_mk'_mul [ContinuousConstSMul Γ T] :
    letI := MulAction.orbitRel Γ T
    IsOpenMap (Quotient.mk' : T → Quotient (MulAction.orbitRel Γ T)) := fun U hU => by
  /-
    Γ : Type u_4
    inst✝³ : Group Γ
    T : Type u_5
    inst✝² : TopologicalSpace T
    inst✝¹ : MulAction Γ T
    inst✝ : ContinuousConstSMul Γ T
    U : Set T
    hU : IsOpen U
    ⊢ IsOpen (Set.image Quotient.mk' U)
  -/
  rw [isOpen_coinduced, MulAction.quotient_preimage_image_eq_union_mul U]
  /-
    Γ : Type u_4
    inst✝³ : Group Γ
    T : Type u_5
    inst✝² : TopologicalSpace T
    inst✝¹ : MulAction Γ T
    inst✝ : ContinuousConstSMul Γ T
    U : Set T
    hU : IsOpen U
    ⊢ IsOpen (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x) U)
  -/
  exact isOpen_iUnion fun γ => isOpenMap_smul γ U hU
  /-
    🎉 no goals
  -/


@[to_additive]
theorem MulAction.isOpenQuotientMap_quotientMk [ContinuousConstSMul Γ T] :
    IsOpenQuotientMap (Quotient.mk (MulAction.orbitRel Γ T)) :=
  ⟨Quot.mk_surjective, continuous_quot_mk, isOpenMap_quotient_mk'_mul⟩


/-- The quotient by a discontinuous group action of a locally compact t2 space is t2. -/
@[to_additive "The quotient by a discontinuous group action of a locally compact t2
space is t2."]
instance (priority := 100) t2Space_of_properlyDiscontinuousSMul_of_t2Space [T2Space T]
    [LocallyCompactSpace T] [ContinuousConstSMul Γ T] [ProperlyDiscontinuousSMul Γ T] :
    T2Space (Quotient (MulAction.orbitRel Γ T)) := by
  /-
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    ⊢ T2Space (Quotient (MulAction.orbitRel Γ T))
  -/
  letI := MulAction.orbitRel Γ T
  /-
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    ⊢ T2Space (Quotient (MulAction.orbitRel Γ T))
  -/
  set Q := Quotient (MulAction.orbitRel Γ T)
  /-
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    ⊢ T2Space (Quotient (MulAction.orbitRel Γ T))
  -/
  rw [t2Space_iff_nhds]
  /-
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    ⊢ Pairwise fun x y => Exists fun U => And (Membership.mem (nhds x) U) (Exists  …
  -/
  let f : T → Q := Quotient.mk'
  /-
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    ⊢ Pairwise fun x y => Exists fun U => And (Membership.mem (nhds x) U) (Exists  …
  -/
  have f_op : IsOpenMap f := isOpenMap_quotient_mk'_mul
  /-
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    ⊢ Pairwise fun x y => Exists fun U => And (Membership.mem (nhds x) U) (Exists  …
  -/
  rintro ⟨x₀⟩ ⟨y₀⟩ (hxy : f x₀ ≠ f y₀)
  /-
    case mk.mk
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    ⊢ Exists fun U => And (Membership.mem (nhds (Quot.mk (⇑(MulAction.orbitRel Γ T …
  -/
  show ∃ U ∈ 𝓝 (f x₀), ∃ V ∈ 𝓝 (f y₀), _
  /-
    case mk.mk
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  have hγx₀y₀ : ∀ γ : Γ, γ • x₀ ≠ y₀ := not_exists.mp (mt Quotient.sound hxy.symm : _)
  /-
    case mk.mk
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  obtain ⟨K₀, hK₀, K₀_in⟩ := exists_compact_mem_nhds x₀
  /-
    case mk.mk.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  obtain ⟨L₀, hL₀, L₀_in⟩ := exists_compact_mem_nhds y₀
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  let bad_Γ_set := { γ : Γ | (γ • ·) '' K₀ ∩ L₀ ≠ ∅ }
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  have bad_Γ_finite : bad_Γ_set.Finite := finite_disjoint_inter_image (Γ := Γ) hK₀ hL₀
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  choose u v hu hv u_v_disjoint using fun γ => t2_separation_nhds (hγx₀y₀ γ)
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  let U₀₀ := ⋂ γ ∈ bad_Γ_set, (γ • ·) ⁻¹' u γ
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  let U₀ := U₀₀ ∩ K₀
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
    U₀ : Set T := Inter.inter U₀₀ K₀
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  let V₀₀ := ⋂ γ ∈ bad_Γ_set, v γ
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
    U₀ : Set T := Inter.inter U₀₀ K₀
    V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  let V₀ := V₀₀ ∩ L₀
  have U_nhds : f '' U₀ ∈ 𝓝 (f x₀) := by
    refine f_op.image_mem_nhds (inter_mem ((biInter_mem bad_Γ_finite).mpr fun γ _ => ?_) K₀_in)
    exact (continuous_const_smul _).continuousAt (hu γ)
  have V_nhds : f '' V₀ ∈ 𝓝 (f y₀) :=
    f_op.image_mem_nhds (inter_mem ((biInter_mem bad_Γ_finite).mpr fun γ _ => hv γ) L₀_in)
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
    U₀ : Set T := Inter.inter U₀₀ K₀
    V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
    V₀ : Set T := Inter.inter V₀₀ L₀
    U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
    V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
    ⊢ Exists fun U => And (Membership.mem (nhds (f x₀)) U) (Exists fun V => And (M …
  -/
  refine ⟨f '' U₀, U_nhds, f '' V₀, V_nhds, MulAction.disjoint_image_image_iff.2 ?_⟩
  /-
    case mk.mk.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
    U₀ : Set T := Inter.inter U₀₀ K₀
    V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
    V₀ : Set T := Inter.inter V₀₀ L₀
    U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
    V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
    ⊢ ∀ (x : T), Membership.mem U₀ x → ∀ (g : Γ), Not (Membership.mem V₀ (HSMul.hS …
  -/
  rintro x ⟨x_in_U₀₀, x_in_K₀⟩ γ
  /-
    case mk.mk.intro.intro.intro.intro.intro
    M : Type u_1
    α : Type u_2
    β : Type u_3
    Γ : Type u_4
    inst✝⁶ : Group Γ
    T : Type u_5
    inst✝⁵ : TopologicalSpace T
    inst✝⁴ : MulAction Γ T
    inst✝³ : T2Space T
    inst✝² : LocallyCompactSpace T
    inst✝¹ : ContinuousConstSMul Γ T
    inst✝ : ProperlyDiscontinuousSMul Γ T
    this : Setoid T := MulAction.orbitRel Γ T
    Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
    f : T → Q := Quotient.mk'
    f_op : IsOpenMap f
    i✝ : Quotient (MulAction.orbitRel Γ T)
    x₀ : T
    j✝ : Quotient (MulAction.orbitRel Γ T)
    y₀ : T
    hxy : Ne (f x₀) (f y₀)
    hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
    K₀ : Set T
    hK₀ : IsCompact K₀
    K₀_in : Membership.mem (nhds x₀) K₀
    L₀ : Set T
    hL₀ : IsCompact L₀
    L₀_in : Membership.mem (nhds y₀) L₀
    bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
    bad_Γ_finite : bad_Γ_set.Finite
    u v : Γ → Set T
    hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
    hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
    u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
    U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
    U₀ : Set T := Inter.inter U₀₀ K₀
    V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
    V₀ : Set T := Inter.inter V₀₀ L₀
    U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
    V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
    x : T
    x_in_U₀₀ : Membership.mem U₀₀ x
    x_in_K₀ : Membership.mem K₀ x
    γ : Γ
    ⊢ Not (Membership.mem V₀ (HSMul.hSMul γ x))
  -/
  by_cases H : γ ∈ bad_Γ_set
    /-
      case pos
      M : Type u_1
      α : Type u_2
      β : Type u_3
      Γ : Type u_4
      inst✝⁶ : Group Γ
      T : Type u_5
      inst✝⁵ : TopologicalSpace T
      inst✝⁴ : MulAction Γ T
      inst✝³ : T2Space T
      inst✝² : LocallyCompactSpace T
      inst✝¹ : ContinuousConstSMul Γ T
      inst✝ : ProperlyDiscontinuousSMul Γ T
      this : Setoid T := MulAction.orbitRel Γ T
      Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
      f : T → Q := Quotient.mk'
      f_op : IsOpenMap f
      i✝ : Quotient (MulAction.orbitRel Γ T)
      x₀ : T
      j✝ : Quotient (MulAction.orbitRel Γ T)
      y₀ : T
      hxy : Ne (f x₀) (f y₀)
      hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
      K₀ : Set T
      hK₀ : IsCompact K₀
      K₀_in : Membership.mem (nhds x₀) K₀
      L₀ : Set T
      hL₀ : IsCompact L₀
      L₀_in : Membership.mem (nhds y₀) L₀
      bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
      bad_Γ_finite : bad_Γ_set.Finite
      u v : Γ → Set T
      hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
      hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
      u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
      U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
      U₀ : Set T := Inter.inter U₀₀ K₀
      V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
      V₀ : Set T := Inter.inter V₀₀ L₀
      U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
      V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
      x : T
      x_in_U₀₀ : Membership.mem U₀₀ x
      x_in_K₀ : Membership.mem K₀ x
      γ : Γ
      H : Membership.mem bad_Γ_set γ
      ⊢ Not (Membership.mem V₀ (HSMul.hSMul γ x))
    -/
  · exact fun h => (u_v_disjoint γ).le_bot ⟨mem_iInter₂.mp x_in_U₀₀ γ H, mem_iInter₂.mp h.1 γ H⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_1
      α : Type u_2
      β : Type u_3
      Γ : Type u_4
      inst✝⁶ : Group Γ
      T : Type u_5
      inst✝⁵ : TopologicalSpace T
      inst✝⁴ : MulAction Γ T
      inst✝³ : T2Space T
      inst✝² : LocallyCompactSpace T
      inst✝¹ : ContinuousConstSMul Γ T
      inst✝ : ProperlyDiscontinuousSMul Γ T
      this : Setoid T := MulAction.orbitRel Γ T
      Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
      f : T → Q := Quotient.mk'
      f_op : IsOpenMap f
      i✝ : Quotient (MulAction.orbitRel Γ T)
      x₀ : T
      j✝ : Quotient (MulAction.orbitRel Γ T)
      y₀ : T
      hxy : Ne (f x₀) (f y₀)
      hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
      K₀ : Set T
      hK₀ : IsCompact K₀
      K₀_in : Membership.mem (nhds x₀) K₀
      L₀ : Set T
      hL₀ : IsCompact L₀
      L₀_in : Membership.mem (nhds y₀) L₀
      bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
      bad_Γ_finite : bad_Γ_set.Finite
      u v : Γ → Set T
      hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
      hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
      u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
      U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
      U₀ : Set T := Inter.inter U₀₀ K₀
      V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
      V₀ : Set T := Inter.inter V₀₀ L₀
      U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
      V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
      x : T
      x_in_U₀₀ : Membership.mem U₀₀ x
      x_in_K₀ : Membership.mem K₀ x
      γ : Γ
      H : Not (Membership.mem bad_Γ_set γ)
      ⊢ Not (Membership.mem V₀ (HSMul.hSMul γ x))
    -/
  · rintro ⟨-, h'⟩
    /-
      case neg.intro
      M : Type u_1
      α : Type u_2
      β : Type u_3
      Γ : Type u_4
      inst✝⁶ : Group Γ
      T : Type u_5
      inst✝⁵ : TopologicalSpace T
      inst✝⁴ : MulAction Γ T
      inst✝³ : T2Space T
      inst✝² : LocallyCompactSpace T
      inst✝¹ : ContinuousConstSMul Γ T
      inst✝ : ProperlyDiscontinuousSMul Γ T
      this : Setoid T := MulAction.orbitRel Γ T
      Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
      f : T → Q := Quotient.mk'
      f_op : IsOpenMap f
      i✝ : Quotient (MulAction.orbitRel Γ T)
      x₀ : T
      j✝ : Quotient (MulAction.orbitRel Γ T)
      y₀ : T
      hxy : Ne (f x₀) (f y₀)
      hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
      K₀ : Set T
      hK₀ : IsCompact K₀
      K₀_in : Membership.mem (nhds x₀) K₀
      L₀ : Set T
      hL₀ : IsCompact L₀
      L₀_in : Membership.mem (nhds y₀) L₀
      bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
      bad_Γ_finite : bad_Γ_set.Finite
      u v : Γ → Set T
      hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
      hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
      u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
      U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
      U₀ : Set T := Inter.inter U₀₀ K₀
      V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
      V₀ : Set T := Inter.inter V₀₀ L₀
      U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
      V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
      x : T
      x_in_U₀₀ : Membership.mem U₀₀ x
      x_in_K₀ : Membership.mem K₀ x
      γ : Γ
      H : Not (Membership.mem bad_Γ_set γ)
      h' : Membership.mem L₀ (HSMul.hSMul γ x)
      ⊢ False
    -/
    simp only [bad_Γ_set, image_smul, Classical.not_not, mem_setOf_eq, Ne] at H
    /-
      case neg.intro
      M : Type u_1
      α : Type u_2
      β : Type u_3
      Γ : Type u_4
      inst✝⁶ : Group Γ
      T : Type u_5
      inst✝⁵ : TopologicalSpace T
      inst✝⁴ : MulAction Γ T
      inst✝³ : T2Space T
      inst✝² : LocallyCompactSpace T
      inst✝¹ : ContinuousConstSMul Γ T
      inst✝ : ProperlyDiscontinuousSMul Γ T
      this : Setoid T := MulAction.orbitRel Γ T
      Q : Type u_5 := Quotient (MulAction.orbitRel Γ T)
      f : T → Q := Quotient.mk'
      f_op : IsOpenMap f
      i✝ : Quotient (MulAction.orbitRel Γ T)
      x₀ : T
      j✝ : Quotient (MulAction.orbitRel Γ T)
      y₀ : T
      hxy : Ne (f x₀) (f y₀)
      hγx₀y₀ : ∀ (γ : Γ), Ne (HSMul.hSMul γ x₀) y₀
      K₀ : Set T
      hK₀ : IsCompact K₀
      K₀_in : Membership.mem (nhds x₀) K₀
      L₀ : Set T
      hL₀ : IsCompact L₀
      L₀_in : Membership.mem (nhds y₀) L₀
      bad_Γ_set : Set Γ := setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul …
      bad_Γ_finite : bad_Γ_set.Finite
      u v : Γ → Set T
      hu : ∀ (γ : Γ), Membership.mem (nhds (HSMul.hSMul γ x₀)) (u γ)
      hv : ∀ (γ : Γ), Membership.mem (nhds y₀) (v γ)
      u_v_disjoint : ∀ (γ : Γ), Disjoint (u γ) (v γ)
      U₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => Set.preimage (fun x =>  …
      U₀ : Set T := Inter.inter U₀₀ K₀
      V₀₀ : Set T := Set.iInter fun γ => Set.iInter fun h => v γ
      V₀ : Set T := Inter.inter V₀₀ L₀
      U_nhds : Membership.mem (nhds (f x₀)) (Set.image f U₀)
      V_nhds : Membership.mem (nhds (f y₀)) (Set.image f V₀)
      x : T
      x_in_U₀₀ : Membership.mem U₀₀ x
      x_in_K₀ : Membership.mem K₀ x
      γ : Γ
      h' : Membership.mem L₀ (HSMul.hSMul γ x)
      H : Eq (Inter.inter (HSMul.hSMul γ K₀) L₀) EmptyCollection.emptyCollection
      ⊢ False
    -/
    exact eq_empty_iff_forall_not_mem.mp H (γ • x) ⟨mem_image_of_mem _ x_in_K₀, h'⟩
    /-
      🎉 no goals
    -/


/-- The quotient of a second countable space by a group action is second countable. -/
@[to_additive "The quotient of a second countable space by an additive group action is second
countable."]
theorem ContinuousConstSMul.secondCountableTopology [SecondCountableTopology T]
    [ContinuousConstSMul Γ T] : SecondCountableTopology (Quotient (MulAction.orbitRel Γ T)) :=
  TopologicalSpace.Quotient.secondCountableTopology isOpenMap_quotient_mk'_mul


/-- Scalar multiplication by a nonzero scalar preserves neighborhoods. -/
theorem smul_mem_nhds_smul_iff₀ {c : G₀} {s : Set α} {x : α} (hc : c ≠ 0) :
    c • s ∈ 𝓝 (c • x : α) ↔ s ∈ 𝓝 x :=
  smul_mem_nhds_smul_iff (Units.mk0 c hc)


@[deprecated (since := "2024-08-06")]
alias set_smul_mem_nhds_smul_iff := smul_mem_nhds_smul_iff₀


alias ⟨_, smul_mem_nhds_smul₀⟩ := smul_mem_nhds_smul_iff₀


@[deprecated smul_mem_nhds_smul₀ (since := "2024-08-06")]
theorem set_smul_mem_nhds_smul {c : G₀} {s : Set α} {x : α} (hs : s ∈ 𝓝 x) (hc : c ≠ 0) :
    c • s ∈ 𝓝 (c • x : α) :=
  smul_mem_nhds_smul₀ hc hs


theorem set_smul_mem_nhds_zero_iff {s : Set α} {c : G₀} (hc : c ≠ 0) :
    c • s ∈ 𝓝 (0 : α) ↔ s ∈ 𝓝 (0 : α) := by
  /-
    α : Type u_2
    G₀ : Type u_6
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : AddMonoid α
    inst✝² : DistribMulAction G₀ α
    inst✝¹ : TopologicalSpace α
    inst✝ : ContinuousConstSMul G₀ α
    s : Set α
    c : G₀
    hc : Ne c 0
    ⊢ Iff (Membership.mem (nhds 0) (HSMul.hSMul c s)) (Membership.mem (nhds 0) s)
  -/
  refine Iff.trans ?_ (smul_mem_nhds_smul_iff₀ hc)
  /-
    α : Type u_2
    G₀ : Type u_6
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : AddMonoid α
    inst✝² : DistribMulAction G₀ α
    inst✝¹ : TopologicalSpace α
    inst✝ : ContinuousConstSMul G₀ α
    s : Set α
    c : G₀
    hc : Ne c 0
    ⊢ Iff (Membership.mem (nhds 0) (HSMul.hSMul c s)) (Membership.mem (nhds (HSMul …
  -/
  rw [smul_zero]
  /-
    🎉 no goals
  -/


