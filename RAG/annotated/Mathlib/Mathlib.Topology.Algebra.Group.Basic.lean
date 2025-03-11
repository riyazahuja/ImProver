/-- Multiplication from the left in a topological group as a homeomorphism. -/
@[to_additive "Addition from the left in a topological additive group as a homeomorphism."]
protected def Homeomorph.mulLeft (a : G) : G ≃ₜ G :=
  { Equiv.mulLeft a with
    continuous_toFun := continuous_const.mul continuous_id
    continuous_invFun := continuous_const.mul continuous_id }


@[to_additive (attr := simp)]
theorem Homeomorph.coe_mulLeft (a : G) : ⇑(Homeomorph.mulLeft a) = (a * ·) :=
  rfl


@[to_additive]
theorem Homeomorph.mulLeft_symm (a : G) : (Homeomorph.mulLeft a).symm = Homeomorph.mulLeft a⁻¹ := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    a : G
    ⊢ Eq (Homeomorph.mulLeft a).symm (Homeomorph.mulLeft (Inv.inv a))
  -/
  ext
  /-
    case H
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    a x✝ : G
    ⊢ Eq ((Homeomorph.mulLeft a).symm x✝) ((Homeomorph.mulLeft (Inv.inv a)) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isOpenMap_mul_left (a : G) : IsOpenMap (a * ·) := (Homeomorph.mulLeft a).isOpenMap


@[to_additive IsOpen.left_addCoset]
theorem IsOpen.leftCoset {U : Set G} (h : IsOpen U) (x : G) : IsOpen (x • U) :=
  isOpenMap_mul_left x _ h


@[to_additive]
lemma isClosedMap_mul_left (a : G) : IsClosedMap (a * ·) := (Homeomorph.mulLeft a).isClosedMap


@[to_additive IsClosed.left_addCoset]
theorem IsClosed.leftCoset {U : Set G} (h : IsClosed U) (x : G) : IsClosed (x • U) :=
  isClosedMap_mul_left x _ h


/-- Multiplication from the right in a topological group as a homeomorphism. -/
@[to_additive "Addition from the right in a topological additive group as a homeomorphism."]
protected def Homeomorph.mulRight (a : G) : G ≃ₜ G :=
  { Equiv.mulRight a with
    continuous_toFun := continuous_id.mul continuous_const
    continuous_invFun := continuous_id.mul continuous_const }


@[to_additive (attr := simp)]
lemma Homeomorph.coe_mulRight (a : G) : ⇑(Homeomorph.mulRight a) = (· * a) := rfl


@[to_additive]
theorem Homeomorph.mulRight_symm (a : G) :
    (Homeomorph.mulRight a).symm = Homeomorph.mulRight a⁻¹ := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    a : G
    ⊢ Eq (Homeomorph.mulRight a).symm (Homeomorph.mulRight (Inv.inv a))
  -/
  ext
  /-
    case H
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    a x✝ : G
    ⊢ Eq ((Homeomorph.mulRight a).symm x✝) ((Homeomorph.mulRight (Inv.inv a)) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isOpenMap_mul_right (a : G) : IsOpenMap (· * a) :=
  (Homeomorph.mulRight a).isOpenMap


@[to_additive IsOpen.right_addCoset]
theorem IsOpen.rightCoset {U : Set G} (h : IsOpen U) (x : G) : IsOpen (op x • U) :=
  isOpenMap_mul_right x _ h


@[to_additive]
theorem isClosedMap_mul_right (a : G) : IsClosedMap (· * a) :=
  (Homeomorph.mulRight a).isClosedMap


@[to_additive IsClosed.right_addCoset]
theorem IsClosed.rightCoset {U : Set G} (h : IsClosed U) (x : G) : IsClosed (op x • U) :=
  isClosedMap_mul_right x _ h


@[to_additive]
theorem discreteTopology_of_isOpen_singleton_one (h : IsOpen ({1} : Set G)) :
    DiscreteTopology G := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    h : IsOpen (Singleton.singleton 1)
    ⊢ DiscreteTopology G
  -/
  rw [← singletons_open_iff_discrete]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : ContinuousMul G
    h : IsOpen (Singleton.singleton 1)
    ⊢ ∀ (a : G), IsOpen (Singleton.singleton a)
  -/
  intro g
  suffices {g} = (g⁻¹ * ·) ⁻¹' {1} by
    rw [this]
    exact (continuous_mul_left g⁻¹).isOpen_preimage _ h
  simp only [mul_one, Set.preimage_mul_left_singleton, eq_self_iff_true, inv_inv,
    Set.singleton_eq_singleton_iff]


@[to_additive]
theorem discreteTopology_iff_isOpen_singleton_one : DiscreteTopology G ↔ IsOpen ({1} : Set G) :=
  ⟨fun h => forall_open_iff_discrete.mpr h {1}, discreteTopology_of_isOpen_singleton_one⟩


/-- Basic hypothesis to talk about a topological additive group. A topological additive group
over `M`, for example, is obtained by requiring the instances `AddGroup M` and
`ContinuousAdd M` and `ContinuousNeg M`. -/
class ContinuousNeg (G : Type u) [TopologicalSpace G] [Neg G] : Prop where
  continuous_neg : Continuous fun a : G => -a


/-- Basic hypothesis to talk about a topological group. A topological group over `M`, for example,
is obtained by requiring the instances `Group M` and `ContinuousMul M` and
`ContinuousInv M`. -/
@[to_additive (attr := continuity)]
class ContinuousInv (G : Type u) [TopologicalSpace G] [Inv G] : Prop where
  continuous_inv : Continuous fun a : G => a⁻¹


@[to_additive]
theorem ContinuousInv.induced {α : Type*} {β : Type*} {F : Type*} [FunLike F α β] [Group α]
    [Group β] [MonoidHomClass F α β] [tβ : TopologicalSpace β] [ContinuousInv β] (f : F) :
    @ContinuousInv α (tβ.induced f) _ := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : Group β
    inst✝¹ : MonoidHomClass F α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousInv β
    f : F
    ⊢ ContinuousInv α
  -/
  let _tα := tβ.induced f
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : Group β
    inst✝¹ : MonoidHomClass F α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousInv β
    f : F
    _tα : TopologicalSpace α := TopologicalSpace.induced (⇑f) tβ
    ⊢ ContinuousInv α
  -/
  refine ⟨continuous_induced_rng.2 ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : Group β
    inst✝¹ : MonoidHomClass F α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousInv β
    f : F
    _tα : TopologicalSpace α := TopologicalSpace.induced (⇑f) tβ
    ⊢ Continuous (Function.comp ⇑f fun a => Inv.inv a)
  -/
  simp only [Function.comp_def, map_inv]
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝⁴ : FunLike F α β
    inst✝³ : Group α
    inst✝² : Group β
    inst✝¹ : MonoidHomClass F α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousInv β
    f : F
    _tα : TopologicalSpace α := TopologicalSpace.induced (⇑f) tβ
    ⊢ Continuous fun x => Inv.inv (f x)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem Specializes.inv {x y : G} (h : x ⤳ y) : (x⁻¹) ⤳ (y⁻¹) :=
  h.map continuous_inv


@[to_additive]
protected theorem Inseparable.inv {x y : G} (h : Inseparable x y) : Inseparable (x⁻¹) (y⁻¹) :=
  h.map continuous_inv


@[to_additive]
protected theorem Specializes.zpow {G : Type*} [DivInvMonoid G] [TopologicalSpace G]
    [ContinuousMul G] [ContinuousInv G] {x y : G} (h : x ⤳ y) : ∀ m : ℤ, (x ^ m) ⤳ (y ^ m)
                   /-
                     G : Type u_1
                     inst✝³ : DivInvMonoid G
                     inst✝² : TopologicalSpace G
                     inst✝¹ : ContinuousMul G
                     inst✝ : ContinuousInv G
                     x y : G
                     h : Specializes x y
                     n : Nat
                     ⊢ Specializes (HPow.hPow x (Int.ofNat n)) (HPow.hPow y (Int.ofNat n))
                   -/
  | .ofNat n => by simpa using h.pow n
                   /-
                     🎉 no goals
                   -/
                     /-
                       G : Type u_1
                       inst✝³ : DivInvMonoid G
                       inst✝² : TopologicalSpace G
                       inst✝¹ : ContinuousMul G
                       inst✝ : ContinuousInv G
                       x y : G
                       h : Specializes x y
                       n : Nat
                       ⊢ Specializes (HPow.hPow x (Int.negSucc n)) (HPow.hPow y (Int.negSucc n))
                     -/
  | .negSucc n => by simpa using (h.pow (n + 1)).inv
                     /-
                       🎉 no goals
                     -/


@[to_additive]
protected theorem Inseparable.zpow {G : Type*} [DivInvMonoid G] [TopologicalSpace G]
    [ContinuousMul G] [ContinuousInv G] {x y : G} (h : Inseparable x y) (m : ℤ) :
    Inseparable (x ^ m) (y ^ m) :=
  (h.specializes.zpow m).antisymm (h.specializes'.zpow m)


@[to_additive]
instance : ContinuousInv (ULift G) :=
  ⟨continuous_uLift_up.comp (continuous_inv.comp continuous_uLift_down)⟩


@[to_additive]
theorem continuousOn_inv {s : Set G} : ContinuousOn Inv.inv s :=
  continuous_inv.continuousOn


@[to_additive]
theorem continuousWithinAt_inv {s : Set G} {x : G} : ContinuousWithinAt Inv.inv s x :=
  continuous_inv.continuousWithinAt


@[to_additive]
theorem continuousAt_inv {x : G} : ContinuousAt Inv.inv x :=
  continuous_inv.continuousAt


@[to_additive]
theorem tendsto_inv (a : G) : Tendsto Inv.inv (𝓝 a) (𝓝 a⁻¹) :=
  continuousAt_inv


/-- If a function converges to a value in a multiplicative topological group, then its inverse
converges to the inverse of this value. For the version in normed fields assuming additionally
that the limit is nonzero, use `Tendsto.inv'`. -/
@[to_additive
  "If a function converges to a value in an additive topological group, then its
  negation converges to the negation of this value."]
theorem Filter.Tendsto.inv {f : α → G} {l : Filter α} {y : G} (h : Tendsto f l (𝓝 y)) :
    Tendsto (fun x => (f x)⁻¹) l (𝓝 y⁻¹) :=
  (continuous_inv.tendsto y).comp h


@[to_additive (attr := continuity, fun_prop)]
theorem Continuous.inv (hf : Continuous f) : Continuous fun x => (f x)⁻¹ :=
  continuous_inv.comp hf


@[to_additive (attr := fun_prop)]
theorem ContinuousAt.inv (hf : ContinuousAt f x) : ContinuousAt (fun x => (f x)⁻¹) x :=
  continuousAt_inv.comp hf


@[to_additive (attr := fun_prop)]
theorem ContinuousOn.inv (hf : ContinuousOn f s) : ContinuousOn (fun x => (f x)⁻¹) s :=
  continuous_inv.comp_continuousOn hf


@[to_additive]
theorem ContinuousWithinAt.inv (hf : ContinuousWithinAt f s x) :
    ContinuousWithinAt (fun x => (f x)⁻¹) s x :=
  Filter.Tendsto.inv hf


@[to_additive]
instance OrderDual.instContinuousInv : ContinuousInv Gᵒᵈ := ‹ContinuousInv G›


@[to_additive]
instance Prod.continuousInv [TopologicalSpace H] [Inv H] [ContinuousInv H] :
    ContinuousInv (G × H) :=
  ⟨continuous_inv.fst'.prod_mk continuous_inv.snd'⟩


@[to_additive]
instance Pi.continuousInv {C : ι → Type*} [∀ i, TopologicalSpace (C i)] [∀ i, Inv (C i)]
    [∀ i, ContinuousInv (C i)] : ContinuousInv (∀ i, C i) where
  continuous_inv := continuous_pi fun i => (continuous_apply i).inv


/-- A version of `Pi.continuousInv` for non-dependent functions. It is needed because sometimes
Lean fails to use `Pi.continuousInv` for non-dependent functions. -/
@[to_additive
  "A version of `Pi.continuousNeg` for non-dependent functions. It is needed
  because sometimes Lean fails to use `Pi.continuousNeg` for non-dependent functions."]
instance Pi.has_continuous_inv' : ContinuousInv (ι → G) :=
  Pi.continuousInv


@[to_additive]
instance (priority := 100) continuousInv_of_discreteTopology [TopologicalSpace H] [Inv H]
    [DiscreteTopology H] : ContinuousInv H :=
  ⟨continuous_of_discreteTopology⟩


@[to_additive]
theorem isClosed_setOf_map_inv [Inv G₁] [Inv G₂] [ContinuousInv G₂] :
    IsClosed { f : G₁ → G₂ | ∀ x, f x⁻¹ = (f x)⁻¹ } := by
  /-
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝⁴ : TopologicalSpace G₂
    inst✝³ : T2Space G₂
    inst✝² : Inv G₁
    inst✝¹ : Inv G₂
    inst✝ : ContinuousInv G₂
    ⊢ IsClosed (setOf fun f => ∀ (x : G₁), Eq (f (Inv.inv x)) (Inv.inv (f x)))
  -/
  simp only [setOf_forall]
  /-
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝⁴ : TopologicalSpace G₂
    inst✝³ : T2Space G₂
    inst✝² : Inv G₁
    inst✝¹ : Inv G₂
    inst✝ : ContinuousInv G₂
    ⊢ IsClosed (Set.iInter fun i => setOf fun x => Eq (x (Inv.inv i)) (Inv.inv (x  …
  -/
  exact isClosed_iInter fun i => isClosed_eq (continuous_apply _) (continuous_apply _).inv
  /-
    🎉 no goals
  -/


instance [TopologicalSpace H] [Inv H] [ContinuousInv H] : ContinuousNeg (Additive H) where
  continuous_neg := @continuous_inv H _ _ _


instance [TopologicalSpace H] [Neg H] [ContinuousNeg H] : ContinuousInv (Multiplicative H) where
  continuous_inv := @continuous_neg H _ _ _


@[to_additive]
theorem IsCompact.inv (hs : IsCompact s) : IsCompact s⁻¹ := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : InvolutiveInv G
    inst✝ : ContinuousInv G
    s : Set G
    hs : IsCompact s
    ⊢ IsCompact (Inv.inv s)
  -/
  rw [← image_inv_eq_inv]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : InvolutiveInv G
    inst✝ : ContinuousInv G
    s : Set G
    hs : IsCompact s
    ⊢ IsCompact (Set.image (fun x => Inv.inv x) s)
  -/
  exact hs.image continuous_inv
  /-
    🎉 no goals
  -/


/-- Inversion in a topological group as a homeomorphism. -/
@[to_additive "Negation in a topological group as a homeomorphism."]
protected def Homeomorph.inv (G : Type*) [TopologicalSpace G] [InvolutiveInv G]
    [ContinuousInv G] : G ≃ₜ G :=
  { Equiv.inv G with
    continuous_toFun := continuous_inv
    continuous_invFun := continuous_inv }


@[to_additive (attr := simp)]
lemma Homeomorph.coe_inv {G : Type*} [TopologicalSpace G] [InvolutiveInv G] [ContinuousInv G] :
    ⇑(Homeomorph.inv G) = Inv.inv := rfl


@[to_additive]
theorem nhds_inv (a : G) : 𝓝 a⁻¹ = (𝓝 a)⁻¹ :=
  ((Homeomorph.inv G).map_nhds_eq a).symm


@[to_additive]
theorem isOpenMap_inv : IsOpenMap (Inv.inv : G → G) :=
  (Homeomorph.inv _).isOpenMap


@[to_additive]
theorem isClosedMap_inv : IsClosedMap (Inv.inv : G → G) :=
  (Homeomorph.inv _).isClosedMap


@[to_additive]
theorem IsOpen.inv (hs : IsOpen s) : IsOpen s⁻¹ :=
  hs.preimage continuous_inv


@[to_additive]
theorem IsClosed.inv (hs : IsClosed s) : IsClosed s⁻¹ :=
  hs.preimage continuous_inv


@[to_additive]
theorem inv_closure : ∀ s : Set G, (closure s)⁻¹ = closure s⁻¹ :=
  (Homeomorph.inv G).preimage_closure


@[to_additive (attr := simp)]
lemma continuous_inv_iff : Continuous f⁻¹ ↔ Continuous f := (Homeomorph.inv G).comp_continuous_iff


@[to_additive (attr := simp)]
lemma continuousAt_inv_iff : ContinuousAt f⁻¹ x ↔ ContinuousAt f x :=
  (Homeomorph.inv G).comp_continuousAt_iff _ _


@[to_additive (attr := simp)]
lemma continuousOn_inv_iff : ContinuousOn f⁻¹ s ↔ ContinuousOn f s :=
  (Homeomorph.inv G).comp_continuousOn_iff _ _


@[to_additive] alias ⟨Continuous.of_inv, _⟩ := continuous_inv_iff

@[to_additive] alias ⟨ContinuousAt.of_inv, _⟩ := continuousAt_inv_iff

@[to_additive] alias ⟨ContinuousOn.of_inv, _⟩ := continuousOn_inv_iff


@[to_additive]
theorem continuousInv_sInf {ts : Set (TopologicalSpace G)}
    (h : ∀ t ∈ ts, @ContinuousInv G t _) : @ContinuousInv G (sInf ts) _ :=
  letI := sInf ts
  { continuous_inv :=
      continuous_sInf_rng.2 fun t ht =>
        continuous_sInf_dom ht (@ContinuousInv.continuous_inv G t _ (h t ht)) }


@[to_additive]
theorem continuousInv_iInf {ts' : ι' → TopologicalSpace G}
    (h' : ∀ i, @ContinuousInv G (ts' i) _) : @ContinuousInv G (⨅ i, ts' i) _ := by
  /-
    G : Type w
    ι' : Sort u_1
    inst✝ : Inv G
    ts' : ι' → TopologicalSpace G
    h' : ∀ (i : ι'), ContinuousInv G
    ⊢ ContinuousInv G
  -/
  rw [← sInf_range]
  /-
    G : Type w
    ι' : Sort u_1
    inst✝ : Inv G
    ts' : ι' → TopologicalSpace G
    h' : ∀ (i : ι'), ContinuousInv G
    ⊢ ContinuousInv G
  -/
  exact continuousInv_sInf (Set.forall_mem_range.mpr h')
  /-
    🎉 no goals
  -/


@[to_additive]
theorem continuousInv_inf {t₁ t₂ : TopologicalSpace G} (h₁ : @ContinuousInv G t₁ _)
    (h₂ : @ContinuousInv G t₂ _) : @ContinuousInv G (t₁ ⊓ t₂) _ := by
  /-
    G : Type w
    inst✝ : Inv G
    t₁ t₂ : TopologicalSpace G
    h₁ : ContinuousInv G
    h₂ : ContinuousInv G
    ⊢ ContinuousInv G
  -/
  rw [inf_eq_iInf]
  /-
    G : Type w
    inst✝ : Inv G
    t₁ t₂ : TopologicalSpace G
    h₁ : ContinuousInv G
    h₂ : ContinuousInv G
    ⊢ ContinuousInv G
  -/
  refine continuousInv_iInf fun b => ?_
  /-
    G : Type w
    inst✝ : Inv G
    t₁ t₂ : TopologicalSpace G
    h₁ : ContinuousInv G
    h₂ : ContinuousInv G
    b : Bool
    ⊢ ContinuousInv G
  -/
              /-
                🎉 no goals
              -/
  cases b <;> assumption
              /-
                🎉 no goals
              -/


@[to_additive]
theorem Topology.IsInducing.continuousInv {G H : Type*} [Inv G] [Inv H] [TopologicalSpace G]
    [TopologicalSpace H] [ContinuousInv H] {f : G → H} (hf : IsInducing f)
    (hf_inv : ∀ x, f x⁻¹ = (f x)⁻¹) : ContinuousInv G :=
                             /-
                               G : Type u_1
                               H : Type u_2
                               inst✝⁴ : Inv G
                               inst✝³ : Inv H
                               inst✝² : TopologicalSpace G
                               inst✝¹ : TopologicalSpace H
                               inst✝ : ContinuousInv H
                               f : G → H
                               hf : Topology.IsInducing f
                               hf_inv : ∀ (x : G), Eq (f (Inv.inv x)) (Inv.inv (f x))
                               ⊢ Continuous (Function.comp f fun a => Inv.inv a)
                             -/
  ⟨hf.continuous_iff.2 <| by simpa only [Function.comp_def, hf_inv] using hf.continuous.inv⟩
                             /-
                               🎉 no goals
                             -/


@[deprecated (since := "2024-10-28")] alias Inducing.continuousInv := IsInducing.continuousInv


/-- A topological (additive) group is a group in which the addition and negation operations are
continuous. -/
class TopologicalAddGroup (G : Type u) [TopologicalSpace G] [AddGroup G] extends
  ContinuousAdd G, ContinuousNeg G : Prop


/-- A topological group is a group in which the multiplication and inversion operations are
continuous.

When you declare an instance that does not already have a `UniformSpace` instance,
you should also provide an instance of `UniformSpace` and `UniformGroup` using
`TopologicalGroup.toUniformSpace` and `topologicalCommGroup_isUniform`. -/
-- Porting note: check that these ↑ names exist once they've been ported in the future.
@[to_additive]
class TopologicalGroup (G : Type*) [TopologicalSpace G] [Group G] extends ContinuousMul G,
  ContinuousInv G : Prop


instance ConjAct.units_continuousConstSMul {M} [Monoid M] [TopologicalSpace M]
    [ContinuousMul M] : ContinuousConstSMul (ConjAct Mˣ) M :=
  ⟨fun _ => (continuous_const.mul continuous_id).mul continuous_const⟩


/-- Conjugation is jointly continuous on `G × G` when both `mul` and `inv` are continuous. -/
@[to_additive
  "Conjugation is jointly continuous on `G × G` when both `add` and `neg` are continuous."]
theorem TopologicalGroup.continuous_conj_prod [ContinuousInv G] :
    Continuous fun g : G × G => g.fst * g.snd * g.fst⁻¹ :=
  continuous_mul.mul (continuous_inv.comp continuous_fst)


/-- Conjugation by a fixed element is continuous when `mul` is continuous. -/
@[to_additive (attr := continuity)
  "Conjugation by a fixed element is continuous when `add` is continuous."]
theorem TopologicalGroup.continuous_conj (g : G) : Continuous fun h : G => g * h * g⁻¹ :=
  (continuous_mul_right g⁻¹).comp (continuous_mul_left g)


/-- Conjugation acting on fixed element of the group is continuous when both `mul` and
`inv` are continuous. -/
@[to_additive (attr := continuity)
  "Conjugation acting on fixed element of the additive group is continuous when both
    `add` and `neg` are continuous."]
theorem TopologicalGroup.continuous_conj' [ContinuousInv G] (h : G) :
    Continuous fun g : G => g * h * g⁻¹ :=
  (continuous_mul_right h).mul continuous_inv


instance : TopologicalGroup (ULift G) where


@[to_additive (attr := continuity)]
theorem continuous_zpow : ∀ z : ℤ, Continuous fun a : G => a ^ z
                      /-
                        G : Type w
                        inst✝² : TopologicalSpace G
                        inst✝¹ : Group G
                        inst✝ : TopologicalGroup G
                        n : Nat
                        ⊢ Continuous fun a => HPow.hPow a (Int.ofNat n)
                      -/
  | Int.ofNat n => by simpa using continuous_pow n
                      /-
                        🎉 no goals
                      -/
                        /-
                          G : Type w
                          inst✝² : TopologicalSpace G
                          inst✝¹ : Group G
                          inst✝ : TopologicalGroup G
                          n : Nat
                          ⊢ Continuous fun a => HPow.hPow a (Int.negSucc n)
                        -/
  | Int.negSucc n => by simpa using (continuous_pow (n + 1)).inv
                        /-
                          🎉 no goals
                        -/


instance AddGroup.continuousConstSMul_int {A} [AddGroup A] [TopologicalSpace A]
    [TopologicalAddGroup A] : ContinuousConstSMul ℤ A :=
  ⟨continuous_zsmul⟩


instance AddGroup.continuousSMul_int {A} [AddGroup A] [TopologicalSpace A]
    [TopologicalAddGroup A] : ContinuousSMul ℤ A :=
  ⟨continuous_prod_of_discrete_left.mpr continuous_zsmul⟩


@[to_additive (attr := continuity, fun_prop)]
theorem Continuous.zpow {f : α → G} (h : Continuous f) (z : ℤ) : Continuous fun b => f b ^ z :=
  (continuous_zpow z).comp h


@[to_additive]
theorem continuousOn_zpow {s : Set G} (z : ℤ) : ContinuousOn (fun x => x ^ z) s :=
  (continuous_zpow z).continuousOn


@[to_additive]
theorem continuousAt_zpow (x : G) (z : ℤ) : ContinuousAt (fun x => x ^ z) x :=
  (continuous_zpow z).continuousAt


@[to_additive]
theorem Filter.Tendsto.zpow {α} {l : Filter α} {f : α → G} {x : G} (hf : Tendsto f l (𝓝 x))
    (z : ℤ) : Tendsto (fun x => f x ^ z) l (𝓝 (x ^ z)) :=
  (continuousAt_zpow _ _).tendsto.comp hf


@[to_additive]
theorem ContinuousWithinAt.zpow {f : α → G} {x : α} {s : Set α} (hf : ContinuousWithinAt f s x)
    (z : ℤ) : ContinuousWithinAt (fun x => f x ^ z) s x :=
  Filter.Tendsto.zpow hf z


@[to_additive (attr := fun_prop)]
theorem ContinuousAt.zpow {f : α → G} {x : α} (hf : ContinuousAt f x) (z : ℤ) :
    ContinuousAt (fun x => f x ^ z) x :=
  Filter.Tendsto.zpow hf z


@[to_additive (attr := fun_prop)]
theorem ContinuousOn.zpow {f : α → G} {s : Set α} (hf : ContinuousOn f s) (z : ℤ) :
    ContinuousOn (fun x => f x ^ z) s := fun x hx => (hf x hx).zpow z


@[to_additive]
theorem tendsto_inv_nhdsGT {a : H} : Tendsto Inv.inv (𝓝[>] a) (𝓝[<] a⁻¹) :=
                                       /-
                                         H : Type x
                                         inst✝² : TopologicalSpace H
                                         inst✝¹ : OrderedCommGroup H
                                         inst✝ : ContinuousInv H
                                         a : H
                                         ⊢ Filter.Tendsto (fun a => Inv.inv a) (Filter.principal (Set.Ioi a)) (Filter.p …
                                       -/
  (continuous_inv.tendsto a).inf <| by simp [tendsto_principal_principal]
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Ioi := tendsto_neg_nhdsGT

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Ioi := tendsto_inv_nhdsGT


@[to_additive]
theorem tendsto_inv_nhdsLT {a : H} : Tendsto Inv.inv (𝓝[<] a) (𝓝[>] a⁻¹) :=
                                       /-
                                         H : Type x
                                         inst✝² : TopologicalSpace H
                                         inst✝¹ : OrderedCommGroup H
                                         inst✝ : ContinuousInv H
                                         a : H
                                         ⊢ Filter.Tendsto (fun a => Inv.inv a) (Filter.principal (Set.Iio a)) (Filter.p …
                                       -/
  (continuous_inv.tendsto a).inf <| by simp [tendsto_principal_principal]
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Iio := tendsto_neg_nhdsLT

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Iio := tendsto_inv_nhdsLT


@[to_additive]
theorem tendsto_inv_nhdsGT_inv {a : H} : Tendsto Inv.inv (𝓝[>] a⁻¹) (𝓝[<] a) := by
  /-
    H : Type x
    inst✝² : TopologicalSpace H
    inst✝¹ : OrderedCommGroup H
    inst✝ : ContinuousInv H
    a : H
    ⊢ Filter.Tendsto Inv.inv (nhdsWithin (Inv.inv a) (Set.Ioi (Inv.inv a))) (nhdsW …
  -/
  simpa only [inv_inv] using @tendsto_inv_nhdsGT _ _ _ _ a⁻¹
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Ioi_neg := tendsto_neg_nhdsGT_neg

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Ioi_inv := tendsto_inv_nhdsGT_inv


@[to_additive]
theorem tendsto_inv_nhdsLT_inv {a : H} : Tendsto Inv.inv (𝓝[<] a⁻¹) (𝓝[>] a) := by
  /-
    H : Type x
    inst✝² : TopologicalSpace H
    inst✝¹ : OrderedCommGroup H
    inst✝ : ContinuousInv H
    a : H
    ⊢ Filter.Tendsto Inv.inv (nhdsWithin (Inv.inv a) (Set.Iio (Inv.inv a))) (nhdsW …
  -/
  simpa only [inv_inv] using @tendsto_inv_nhdsLT _ _ _ _ a⁻¹
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Iio_neg := tendsto_neg_nhdsLT_neg

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Iio_inv := tendsto_inv_nhdsLT_inv


@[to_additive]
theorem tendsto_inv_nhdsGE {a : H} : Tendsto Inv.inv (𝓝[≥] a) (𝓝[≤] a⁻¹) :=
                                       /-
                                         H : Type x
                                         inst✝² : TopologicalSpace H
                                         inst✝¹ : OrderedCommGroup H
                                         inst✝ : ContinuousInv H
                                         a : H
                                         ⊢ Filter.Tendsto (fun a => Inv.inv a) (Filter.principal (Set.Ici a)) (Filter.p …
                                       -/
  (continuous_inv.tendsto a).inf <| by simp [tendsto_principal_principal]
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Ici := tendsto_neg_nhdsGE

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Ici := tendsto_inv_nhdsGE


@[to_additive]
theorem tendsto_inv_nhdsLE {a : H} : Tendsto Inv.inv (𝓝[≤] a) (𝓝[≥] a⁻¹) :=
                                       /-
                                         H : Type x
                                         inst✝² : TopologicalSpace H
                                         inst✝¹ : OrderedCommGroup H
                                         inst✝ : ContinuousInv H
                                         a : H
                                         ⊢ Filter.Tendsto (fun a => Inv.inv a) (Filter.principal (Set.Iic a)) (Filter.p …
                                       -/
  (continuous_inv.tendsto a).inf <| by simp [tendsto_principal_principal]
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Iic := tendsto_neg_nhdsLE

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Iic := tendsto_inv_nhdsLE


@[to_additive]
theorem tendsto_inv_nhdsGE_inv {a : H} : Tendsto Inv.inv (𝓝[≥] a⁻¹) (𝓝[≤] a) := by
  /-
    H : Type x
    inst✝² : TopologicalSpace H
    inst✝¹ : OrderedCommGroup H
    inst✝ : ContinuousInv H
    a : H
    ⊢ Filter.Tendsto Inv.inv (nhdsWithin (Inv.inv a) (Set.Ici (Inv.inv a))) (nhdsW …
  -/
  simpa only [inv_inv] using @tendsto_inv_nhdsGE _ _ _ _ a⁻¹
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Ici_neg := tendsto_neg_nhdsGE_neg

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Ici_inv := tendsto_inv_nhdsGE_inv


@[to_additive]
theorem tendsto_inv_nhdsLE_inv {a : H} : Tendsto Inv.inv (𝓝[≤] a⁻¹) (𝓝[≥] a) := by
  /-
    H : Type x
    inst✝² : TopologicalSpace H
    inst✝¹ : OrderedCommGroup H
    inst✝ : ContinuousInv H
    a : H
    ⊢ Filter.Tendsto Inv.inv (nhdsWithin (Inv.inv a) (Set.Iic (Inv.inv a))) (nhdsW …
  -/
  simpa only [inv_inv] using @tendsto_inv_nhdsLE _ _ _ _ a⁻¹
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias tendsto_neg_nhdsWithin_Iic_neg := tendsto_neg_nhdsLE_neg

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_inv_nhdsWithin_Iic_inv := tendsto_inv_nhdsLE_inv


@[to_additive]
instance [TopologicalSpace H] [Group H] [TopologicalGroup H] : TopologicalGroup (G × H) where
  continuous_inv := continuous_inv.prodMap continuous_inv


@[to_additive]
instance OrderDual.instTopologicalGroup : TopologicalGroup Gᵒᵈ where


@[to_additive]
instance Pi.topologicalGroup {C : β → Type*} [∀ b, TopologicalSpace (C b)] [∀ b, Group (C b)]
    [∀ b, TopologicalGroup (C b)] : TopologicalGroup (∀ b, C b) where
  continuous_inv := continuous_pi fun i => (continuous_apply i).inv


@[to_additive]
instance [Inv α] [ContinuousInv α] : ContinuousInv αᵐᵒᵖ :=
  opHomeomorph.symm.isInducing.continuousInv unop_inv


/-- If multiplication is continuous in `α`, then it also is in `αᵐᵒᵖ`. -/
@[to_additive "If addition is continuous in `α`, then it also is in `αᵃᵒᵖ`."]
instance [Group α] [TopologicalGroup α] : TopologicalGroup αᵐᵒᵖ where


@[to_additive]
theorem nhds_one_symm : comap Inv.inv (𝓝 (1 : G)) = 𝓝 (1 : G) :=
  ((Homeomorph.inv G).comap_nhds_eq _).trans (congr_arg nhds inv_one)


@[to_additive]
theorem nhds_one_symm' : map Inv.inv (𝓝 (1 : G)) = 𝓝 (1 : G) :=
  ((Homeomorph.inv G).map_nhds_eq _).trans (congr_arg nhds inv_one)


@[to_additive]
theorem inv_mem_nhds_one {S : Set G} (hS : S ∈ (𝓝 1 : Filter G)) : S⁻¹ ∈ 𝓝 (1 : G) := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    S : Set G
    hS : Membership.mem (nhds 1) S
    ⊢ Membership.mem (nhds 1) (Inv.inv S)
  -/
  rwa [← nhds_one_symm'] at hS
  /-
    🎉 no goals
  -/


/-- The map `(x, y) ↦ (x, x * y)` as a homeomorphism. This is a shear mapping. -/
@[to_additive "The map `(x, y) ↦ (x, x + y)` as a homeomorphism. This is a shear mapping."]
protected def Homeomorph.shearMulRight : G × G ≃ₜ G × G :=
  { Equiv.prodShear (Equiv.refl _) Equiv.mulLeft with
    continuous_toFun := continuous_fst.prod_mk continuous_mul
    continuous_invFun := continuous_fst.prod_mk <| continuous_fst.inv.mul continuous_snd }


@[to_additive (attr := simp)]
theorem Homeomorph.shearMulRight_coe :
    ⇑(Homeomorph.shearMulRight G) = fun z : G × G => (z.1, z.1 * z.2) :=
  rfl


@[to_additive (attr := simp)]
theorem Homeomorph.shearMulRight_symm_coe :
    ⇑(Homeomorph.shearMulRight G).symm = fun z : G × G => (z.1, z.1⁻¹ * z.2) :=
  rfl


@[to_additive]
protected theorem Topology.IsInducing.topologicalGroup {F : Type*} [Group H] [TopologicalSpace H]
    [FunLike F H G] [MonoidHomClass F H G] (f : F) (hf : IsInducing f) : TopologicalGroup H :=
  { toContinuousMul := hf.continuousMul _
    toContinuousInv := hf.continuousInv (map_inv f) }


@[deprecated (since := "2024-10-28")] alias Inducing.topologicalGroup := IsInducing.topologicalGroup


@[to_additive]
theorem topologicalGroup_induced {F : Type*} [Group H] [FunLike F H G] [MonoidHomClass F H G]
    (f : F) :
    @TopologicalGroup H (induced f ‹_›) _ :=
  letI := induced f ‹_›
  IsInducing.topologicalGroup f ⟨rfl⟩


@[to_additive]
instance (S : Subgroup G) : TopologicalGroup S :=
  IsInducing.subtypeVal.topologicalGroup S.subtype


/-- The (topological-space) closure of a subgroup of a topological group is
itself a subgroup. -/
@[to_additive
  "The (topological-space) closure of an additive subgroup of an additive topological group is
  itself an additive subgroup."]
def Subgroup.topologicalClosure (s : Subgroup G) : Subgroup G :=
  { s.toSubmonoid.topologicalClosure with
    carrier := _root_.closure (s : Set G)
                                 /-
                                   G : Type w
                                   H : Type x
                                   α : Type u
                                   β : Type v
                                   inst✝³ : TopologicalSpace G
                                   inst✝² : Group G
                                   inst✝¹ : TopologicalGroup G
                                   inst✝ : TopologicalSpace α
                                   f : α → G
                                   s✝ : Set α
                                   x : α
                                   s : Subgroup G
                                   g : G
                                   hg : Membership.mem { carrier := _root_.closure ↑s, mul_mem' := ⋯, one_mem' := …
                                   ⊢ Membership.mem { carrier := _root_.closure ↑s, mul_mem' := ⋯, one_mem' := ⋯  …
                                 -/
    inv_mem' := fun {g} hg => by simpa only [← Set.mem_inv, inv_closure, inv_coe_set] using hg }
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive (attr := simp)]
theorem Subgroup.topologicalClosure_coe {s : Subgroup G} :
    (s.topologicalClosure : Set G) = _root_.closure s :=
  rfl


@[to_additive]
theorem Subgroup.le_topologicalClosure (s : Subgroup G) : s ≤ s.topologicalClosure :=
  _root_.subset_closure


@[to_additive]
theorem Subgroup.isClosed_topologicalClosure (s : Subgroup G) :
    IsClosed (s.topologicalClosure : Set G) := isClosed_closure


@[to_additive]
theorem Subgroup.topologicalClosure_minimal (s : Subgroup G) {t : Subgroup G} (h : s ≤ t)
    (ht : IsClosed (t : Set G)) : s.topologicalClosure ≤ t :=
  closure_minimal h ht


@[to_additive]
theorem DenseRange.topologicalClosure_map_subgroup [Group H] [TopologicalSpace H]
    [TopologicalGroup H] {f : G →* H} (hf : Continuous f) (hf' : DenseRange f) {s : Subgroup G}
    (hs : s.topologicalClosure = ⊤) : (s.map f).topologicalClosure = ⊤ := by
  /-
    G : Type w
    H : Type x
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    hf : Continuous ⇑f
    hf' : DenseRange ⇑f
    s : Subgroup G
    hs : Eq s.topologicalClosure Top.top
    ⊢ Eq (Subgroup.map f s).topologicalClosure Top.top
  -/
  rw [SetLike.ext'_iff] at hs ⊢
  /-
    G : Type w
    H : Type x
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    hf : Continuous ⇑f
    hf' : DenseRange ⇑f
    s : Subgroup G
    hs : Eq ↑s.topologicalClosure ↑Top.top
    ⊢ Eq ↑(Subgroup.map f s).topologicalClosure ↑Top.top
  -/
  simp only [Subgroup.topologicalClosure_coe, Subgroup.coe_top, ← dense_iff_closure_eq] at hs ⊢
  /-
    G : Type w
    H : Type x
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    hf : Continuous ⇑f
    hf' : DenseRange ⇑f
    s : Subgroup G
    hs : Dense ↑s
    ⊢ Dense ↑(Subgroup.map f s)
  -/
  exact hf'.dense_image hf hs
  /-
    🎉 no goals
  -/


/-- The topological closure of a normal subgroup is normal. -/
@[to_additive "The topological closure of a normal additive subgroup is normal."]
theorem Subgroup.is_normal_topologicalClosure {G : Type*} [TopologicalSpace G] [Group G]
    [TopologicalGroup G] (N : Subgroup G) [N.Normal] : (Subgroup.topologicalClosure N).Normal where
  conj_mem n hn g := by
    /-
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      N : Subgroup G
      inst✝ : N.Normal
      n : G
      hn : Membership.mem N.topologicalClosure n
      g : G
      ⊢ Membership.mem N.topologicalClosure (HMul.hMul (HMul.hMul g n) (Inv.inv g))
    -/
    apply map_mem_closure (TopologicalGroup.continuous_conj g) hn
    /-
      G : Type u_1
      inst✝³ : TopologicalSpace G
      inst✝² : Group G
      inst✝¹ : TopologicalGroup G
      N : Subgroup G
      inst✝ : N.Normal
      n : G
      hn : Membership.mem N.topologicalClosure n
      g : G
      ⊢ Set.MapsTo (fun h => HMul.hMul (HMul.hMul g h) (Inv.inv g)) ↑N ↑N
    -/
    exact fun m hm => Subgroup.Normal.conj_mem inferInstance m hm g
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mul_mem_connectedComponent_one {G : Type*} [TopologicalSpace G] [MulOneClass G]
    [ContinuousMul G] {g h : G} (hg : g ∈ connectedComponent (1 : G))
    (hh : h ∈ connectedComponent (1 : G)) : g * h ∈ connectedComponent (1 : G) := by
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : MulOneClass G
    inst✝ : ContinuousMul G
    g h : G
    hg : Membership.mem (connectedComponent 1) g
    hh : Membership.mem (connectedComponent 1) h
    ⊢ Membership.mem (connectedComponent 1) (HMul.hMul g h)
  -/
  rw [connectedComponent_eq hg]
  have hmul : g ∈ connectedComponent (g * h) := by
    apply Continuous.image_connectedComponent_subset (continuous_mul_left g)
    rw [← connectedComponent_eq hh]
    exact ⟨(1 : G), mem_connectedComponent, by simp only [mul_one]⟩
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : MulOneClass G
    inst✝ : ContinuousMul G
    g h : G
    hg : Membership.mem (connectedComponent 1) g
    hh : Membership.mem (connectedComponent 1) h
    hmul : Membership.mem (connectedComponent (HMul.hMul g h)) g
    ⊢ Membership.mem (connectedComponent g) (HMul.hMul g h)
  -/
  simpa [← connectedComponent_eq hmul] using mem_connectedComponent
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inv_mem_connectedComponent_one {G : Type*} [TopologicalSpace G] [Group G]
    [TopologicalGroup G] {g : G} (hg : g ∈ connectedComponent (1 : G)) :
    g⁻¹ ∈ connectedComponent (1 : G) := by
  /-
    G : Type u_1
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    g : G
    hg : Membership.mem (connectedComponent 1) g
    ⊢ Membership.mem (connectedComponent 1) (Inv.inv g)
  -/
  rw [← inv_one]
  exact
    Continuous.image_connectedComponent_subset continuous_inv _
      ((Set.mem_image _ _ _).mp ⟨g, hg, rfl⟩)


/-- The connected component of 1 is a subgroup of `G`. -/
@[to_additive "The connected component of 0 is a subgroup of `G`."]
def Subgroup.connectedComponentOfOne (G : Type*) [TopologicalSpace G] [Group G]
    [TopologicalGroup G] : Subgroup G where
  carrier := connectedComponent (1 : G)
  one_mem' := mem_connectedComponent
  mul_mem' hg hh := mul_mem_connectedComponent_one hg hh
  inv_mem' hg := inv_mem_connectedComponent_one hg


/-- If a subgroup of a topological group is commutative, then so is its topological closure.

See note [reducible non-instances]. -/
@[to_additive
  "If a subgroup of an additive topological group is commutative, then so is its
topological closure.

See note [reducible non-instances]."]
abbrev Subgroup.commGroupTopologicalClosure [T2Space G] (s : Subgroup G)
    (hs : ∀ x y : s, x * y = y * x) : CommGroup s.topologicalClosure :=
  { s.topologicalClosure.toGroup, s.toSubmonoid.commMonoidTopologicalClosure hs with }


variable (G) in
@[to_additive]
lemma Subgroup.coe_topologicalClosure_bot :
                                                                                       /-
                                                                                         G : Type w
                                                                                         inst✝² : TopologicalSpace G
                                                                                         inst✝¹ : Group G
                                                                                         inst✝ : TopologicalGroup G
                                                                                         ⊢ Eq (↑Bot.bot.topologicalClosure) (_root_.closure (Singleton.singleton 1))
                                                                                       -/
    ((⊥ : Subgroup G).topologicalClosure : Set G) = _root_.closure ({1} : Set G) := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[to_additive exists_nhds_half_neg]
theorem exists_nhds_split_inv {s : Set G} (hs : s ∈ 𝓝 (1 : G)) :
    ∃ V ∈ 𝓝 (1 : G), ∀ v ∈ V, ∀ w ∈ V, v / w ∈ s := by
  have : (fun p : G × G => p.1 * p.2⁻¹) ⁻¹' s ∈ 𝓝 ((1, 1) : G × G) :=
    continuousAt_fst.mul continuousAt_snd.inv (by simpa)
  simpa only [div_eq_mul_inv, nhds_prod_eq, mem_prod_self_iff, prod_subset_iff, mem_preimage] using
    this


@[to_additive]
theorem nhds_translation_mul_inv (x : G) : comap (· * x⁻¹) (𝓝 1) = 𝓝 x :=
                                                                                   /-
                                                                                     G : Type w
                                                                                     inst✝² : TopologicalSpace G
                                                                                     inst✝¹ : Group G
                                                                                     inst✝ : TopologicalGroup G
                                                                                     x : G
                                                                                     ⊢ Eq (nhds (HMul.hMul 1 (Inv.inv (Inv.inv x)))) (nhds x)
                                                                                   -/
  ((Homeomorph.mulRight x⁻¹).comap_nhds_eq 1).trans <| show 𝓝 (1 * x⁻¹⁻¹) = 𝓝 x by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[to_additive (attr := simp)]
theorem map_mul_left_nhds (x y : G) : map (x * ·) (𝓝 y) = 𝓝 (x * y) :=
  (Homeomorph.mulLeft x).map_nhds_eq y


@[to_additive]
                                                                      /-
                                                                        G : Type w
                                                                        inst✝² : TopologicalSpace G
                                                                        inst✝¹ : Group G
                                                                        inst✝ : TopologicalGroup G
                                                                        x : G
                                                                        ⊢ Eq (Filter.map (fun x_1 => HMul.hMul x x_1) (nhds 1)) (nhds x)
                                                                      -/
theorem map_mul_left_nhds_one (x : G) : map (x * ·) (𝓝 1) = 𝓝 x := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive (attr := simp)]
theorem map_mul_right_nhds (x y : G) : map (· * x) (𝓝 y) = 𝓝 (y * x) :=
  (Homeomorph.mulRight x).map_nhds_eq y


@[to_additive]
                                                                       /-
                                                                         G : Type w
                                                                         inst✝² : TopologicalSpace G
                                                                         inst✝¹ : Group G
                                                                         inst✝ : TopologicalGroup G
                                                                         x : G
                                                                         ⊢ Eq (Filter.map (fun x_1 => HMul.hMul x_1 x) (nhds 1)) (nhds x)
                                                                       -/
theorem map_mul_right_nhds_one (x : G) : map (· * x) (𝓝 1) = 𝓝 x := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive]
theorem Filter.HasBasis.nhds_of_one {ι : Sort*} {p : ι → Prop} {s : ι → Set G}
    (hb : HasBasis (𝓝 1 : Filter G) p s) (x : G) :
    HasBasis (𝓝 x) p fun i => { y | y / x ∈ s i } := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    ι : Sort u_1
    p : ι → Prop
    s : ι → Set G
    hb : (nhds 1).HasBasis p s
    x : G
    ⊢ (nhds x).HasBasis p fun i => setOf fun y => Membership.mem (s i) (HDiv.hDiv  …
  -/
  rw [← nhds_translation_mul_inv]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    ι : Sort u_1
    p : ι → Prop
    s : ι → Set G
    hb : (nhds 1).HasBasis p s
    x : G
    ⊢ (Filter.comap (fun x_1 => HMul.hMul x_1 (Inv.inv x)) (nhds 1)).HasBasis p fu …
  -/
  simp_rw [div_eq_mul_inv]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    ι : Sort u_1
    p : ι → Prop
    s : ι → Set G
    hb : (nhds 1).HasBasis p s
    x : G
    ⊢ (Filter.comap (fun x_1 => HMul.hMul x_1 (Inv.inv x)) (nhds 1)).HasBasis p fu …
  -/
  exact hb.comap _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_closure_iff_nhds_one {x : G} {s : Set G} :
    x ∈ closure s ↔ ∀ U ∈ (𝓝 1 : Filter G), ∃ y ∈ s, y / x ∈ U := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    x : G
    s : Set G
    ⊢ Iff (Membership.mem (closure s) x) (∀ (U : Set G), Membership.mem (nhds 1) U …
  -/
  rw [mem_closure_iff_nhds_basis ((𝓝 1 : Filter G).basis_sets.nhds_of_one x)]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    x : G
    s : Set G
    ⊢ Iff (∀ (i : Set G), Membership.mem (nhds 1) i → Exists fun y => And (Members …
  -/
  simp_rw [Set.mem_setOf, id]
  /-
    🎉 no goals
  -/


/-- A monoid homomorphism (a bundled morphism of a type that implements `MonoidHomClass`) from a
topological group to a topological monoid is continuous provided that it is continuous at one. See
also `uniformContinuous_of_continuousAt_one`. -/
@[to_additive
  "An additive monoid homomorphism (a bundled morphism of a type that implements
  `AddMonoidHomClass`) from an additive topological group to an additive topological monoid is
  continuous provided that it is continuous at zero. See also
  `uniformContinuous_of_continuousAt_zero`."]
theorem continuous_of_continuousAt_one {M hom : Type*} [MulOneClass M] [TopologicalSpace M]
    [ContinuousMul M] [FunLike hom G M] [MonoidHomClass hom G M] (f : hom)
    (hf : ContinuousAt f 1) :
    Continuous f :=
  continuous_iff_continuousAt.2 fun x => by
    simpa only [ContinuousAt, ← map_mul_left_nhds_one x, tendsto_map'_iff, Function.comp_def,
      map_mul, map_one, mul_one] using hf.tendsto.const_mul (f x)


@[to_additive continuous_of_continuousAt_zero₂]
theorem continuous_of_continuousAt_one₂ {H M : Type*} [CommMonoid M] [TopologicalSpace M]
    [ContinuousMul M] [Group H] [TopologicalSpace H] [TopologicalGroup H] (f : G →* H →* M)
    (hf : ContinuousAt (fun x : G × H ↦ f x.1 x.2) (1, 1))
    (hl : ∀ x, ContinuousAt (f x) 1) (hr : ∀ y, ContinuousAt (f · y) 1) :
    Continuous (fun x : G × H ↦ f x.1 x.2) := continuous_iff_continuousAt.2 fun (x, y) => by
  simp only [ContinuousAt, nhds_prod_eq, ← map_mul_left_nhds_one x, ← map_mul_left_nhds_one y,
    prod_map_map_eq, tendsto_map'_iff, Function.comp_def, map_mul, MonoidHom.mul_apply] at *
  refine ((tendsto_const_nhds.mul ((hr y).comp tendsto_fst)).mul
    (((hl x).comp tendsto_snd).mul hf)).mono_right (le_of_eq ?_)
  /-
    G : Type w
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : Group G
    inst✝⁶ : TopologicalGroup G
    H : Type u_1
    M : Type u_2
    inst✝⁵ : CommMonoid M
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ContinuousMul M
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G (MonoidHom H M)
    hl : ∀ (x : G), Filter.Tendsto (⇑(f x)) (nhds 1) (nhds ((f x) 1))
    hr : ∀ (y : H), Filter.Tendsto (fun x => (f x) y) (nhds 1) (nhds ((f 1) y))
    x✝ : Prod G H
    x : G
    y : H
    hf : Filter.Tendsto (fun x => (f x.1) x.2) (SProd.sprod (nhds 1) (nhds 1)) (nh …
    ⊢ Eq (nhds (HMul.hMul (HMul.hMul ((f x) y) ((f 1) y)) (HMul.hMul ((f x) 1) ((f …
  -/
  simp only [map_one, mul_one, MonoidHom.one_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma TopologicalGroup.isInducing_iff_nhds_one
    {H : Type*} [Group H] [TopologicalSpace H] [TopologicalGroup H] {f : G →* H} :
    Topology.IsInducing f ↔ 𝓝 (1 : G) = (𝓝 (1 : H)).comap f := by
  /-
    G : Type w
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    H : Type u_1
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    ⊢ Iff (Topology.IsInducing ⇑f) (Eq (nhds 1) (Filter.comap (⇑f) (nhds 1)))
  -/
  rw [Topology.isInducing_iff_nhds]
  /-
    G : Type w
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    H : Type u_1
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    ⊢ Iff (∀ (x : G), Eq (nhds x) (Filter.comap (⇑f) (nhds (f x)))) (Eq (nhds 1) ( …
  -/
  refine ⟨(f.map_one ▸ · 1), fun hf x ↦ ?_⟩
  rw [← nhds_translation_mul_inv, ← nhds_translation_mul_inv (f x), Filter.comap_comap, hf,
    Filter.comap_comap]
  /-
    G : Type w
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    H : Type u_1
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    hf : Eq (nhds 1) (Filter.comap (⇑f) (nhds 1))
    x : G
    ⊢ Eq (Filter.comap (Function.comp ⇑f fun x_1 => HMul.hMul x_1 (Inv.inv x)) (nh …
  -/
  congr 1
  /-
    case e_m
    G : Type w
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    H : Type u_1
    inst✝² : Group H
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalGroup H
    f : MonoidHom G H
    hf : Eq (nhds 1) (Filter.comap (⇑f) (nhds 1))
    x : G
    ⊢ Eq (Function.comp ⇑f fun x_1 => HMul.hMul x_1 (Inv.inv x)) (Function.comp (f …
  -/
  ext; simp
       /-
         🎉 no goals
       -/

-- TODO: unify with `QuotientGroup.isOpenQuotientMap_mk`

/-- Let `A` and `B` be topological groups, and let `φ : A → B` be a continuous surjective group
homomorphism. Assume furthermore that `φ` is a quotient map (i.e., `V ⊆ B`
is open iff `φ⁻¹ V` is open). Then `φ` is an open quotient map, and in particular an open map. -/
@[to_additive "Let `A` and `B` be topological additive groups, and let `φ : A → B` be a continuous
surjective additive group homomorphism. Assume furthermore that `φ` is a quotient map (i.e., `V ⊆ B`
is open iff `φ⁻¹ V` is open). Then `φ` is an open quotient map, and in particular an open map."]
lemma MonoidHom.isOpenQuotientMap_of_isQuotientMap {A : Type*} [Group A]
    [TopologicalSpace A] [TopologicalGroup A] {B : Type*} [Group B] [TopologicalSpace B]
    {F : Type*} [FunLike F A B] [MonoidHomClass F A B] {φ : F}
    (hφ : IsQuotientMap φ) : IsOpenQuotientMap φ where
    surjective := hφ.surjective
    continuous := hφ.continuous
    isOpenMap := by
      -- We need to check that if `U ⊆ A` is open then `φ⁻¹ (φ U)` is open.
      /-
        A : Type u_1
        inst✝⁶ : Group A
        inst✝⁵ : TopologicalSpace A
        inst✝⁴ : TopologicalGroup A
        B : Type u_2
        inst✝³ : Group B
        inst✝² : TopologicalSpace B
        F : Type u_3
        inst✝¹ : FunLike F A B
        inst✝ : MonoidHomClass F A B
        φ : F
        hφ : Topology.IsQuotientMap ⇑φ
        ⊢ IsOpenMap ⇑φ
      -/
      intro U hU
      /-
        A : Type u_1
        inst✝⁶ : Group A
        inst✝⁵ : TopologicalSpace A
        inst✝⁴ : TopologicalGroup A
        B : Type u_2
        inst✝³ : Group B
        inst✝² : TopologicalSpace B
        F : Type u_3
        inst✝¹ : FunLike F A B
        inst✝ : MonoidHomClass F A B
        φ : F
        hφ : Topology.IsQuotientMap ⇑φ
        U : Set A
        hU : IsOpen U
        ⊢ IsOpen (Set.image (⇑φ) U)
      -/
      rw [← hφ.isOpen_preimage]
      -- It suffices to show that `φ⁻¹ (φ U) = ⋃ (U * k⁻¹)` as `k` runs through the kernel of `φ`,
      -- as `U * k⁻¹` is open because `x ↦ x * k` is continuous.
      -- Remark: here is where we use that we have groups not monoids (you cannot avoid
      -- using both `k` and `k⁻¹` at this point).
      suffices ⇑φ ⁻¹' (⇑φ '' U) = ⋃ k ∈ ker (φ : A →* B), (fun x ↦ x * k) ⁻¹' U by
        exact this ▸ isOpen_biUnion (fun k _ ↦ Continuous.isOpen_preimage (by fun_prop) _ hU)
      /-
        A : Type u_1
        inst✝⁶ : Group A
        inst✝⁵ : TopologicalSpace A
        inst✝⁴ : TopologicalGroup A
        B : Type u_2
        inst✝³ : Group B
        inst✝² : TopologicalSpace B
        F : Type u_3
        inst✝¹ : FunLike F A B
        inst✝ : MonoidHomClass F A B
        φ : F
        hφ : Topology.IsQuotientMap ⇑φ
        U : Set A
        hU : IsOpen U
        ⊢ Eq (Set.preimage (⇑φ) (Set.image (⇑φ) U)) (Set.iUnion fun k => Set.iUnion fu …
      -/
      ext x
      -- But this is an elementary calculation.
      /-
        case h
        A : Type u_1
        inst✝⁶ : Group A
        inst✝⁵ : TopologicalSpace A
        inst✝⁴ : TopologicalGroup A
        B : Type u_2
        inst✝³ : Group B
        inst✝² : TopologicalSpace B
        F : Type u_3
        inst✝¹ : FunLike F A B
        inst✝ : MonoidHomClass F A B
        φ : F
        hφ : Topology.IsQuotientMap ⇑φ
        U : Set A
        hU : IsOpen U
        x : A
        ⊢ Iff (Membership.mem (Set.preimage (⇑φ) (Set.image (⇑φ) U)) x) (Membership.me …
      -/
      constructor
        /-
          case h.mp
          A : Type u_1
          inst✝⁶ : Group A
          inst✝⁵ : TopologicalSpace A
          inst✝⁴ : TopologicalGroup A
          B : Type u_2
          inst✝³ : Group B
          inst✝² : TopologicalSpace B
          F : Type u_3
          inst✝¹ : FunLike F A B
          inst✝ : MonoidHomClass F A B
          φ : F
          hφ : Topology.IsQuotientMap ⇑φ
          U : Set A
          hU : IsOpen U
          x : A
          ⊢ Membership.mem (Set.preimage (⇑φ) (Set.image (⇑φ) U)) x → Membership.mem (Se …
        -/
      · rintro ⟨y, hyU, hyx⟩
        /-
          case h.mp.intro.intro
          A : Type u_1
          inst✝⁶ : Group A
          inst✝⁵ : TopologicalSpace A
          inst✝⁴ : TopologicalGroup A
          B : Type u_2
          inst✝³ : Group B
          inst✝² : TopologicalSpace B
          F : Type u_3
          inst✝¹ : FunLike F A B
          inst✝ : MonoidHomClass F A B
          φ : F
          hφ : Topology.IsQuotientMap ⇑φ
          U : Set A
          hU : IsOpen U
          x y : A
          hyU : Membership.mem U y
          hyx : Eq (φ y) (φ x)
          ⊢ Membership.mem (Set.iUnion fun k => Set.iUnion fun h => Set.preimage (fun x  …
        -/
        apply Set.mem_iUnion_of_mem (x⁻¹ * y)
        /-
          case h.mp.intro.intro
          A : Type u_1
          inst✝⁶ : Group A
          inst✝⁵ : TopologicalSpace A
          inst✝⁴ : TopologicalGroup A
          B : Type u_2
          inst✝³ : Group B
          inst✝² : TopologicalSpace B
          F : Type u_3
          inst✝¹ : FunLike F A B
          inst✝ : MonoidHomClass F A B
          φ : F
          hφ : Topology.IsQuotientMap ⇑φ
          U : Set A
          hU : IsOpen U
          x y : A
          hyU : Membership.mem U y
          hyx : Eq (φ y) (φ x)
          ⊢ Membership.mem (Set.iUnion fun h => Set.preimage (fun x_1 => HMul.hMul x_1 ( …
        -/
        simp_all
        /-
          🎉 no goals
        -/
        /-
          case h.mpr
          A : Type u_1
          inst✝⁶ : Group A
          inst✝⁵ : TopologicalSpace A
          inst✝⁴ : TopologicalGroup A
          B : Type u_2
          inst✝³ : Group B
          inst✝² : TopologicalSpace B
          F : Type u_3
          inst✝¹ : FunLike F A B
          inst✝ : MonoidHomClass F A B
          φ : F
          hφ : Topology.IsQuotientMap ⇑φ
          U : Set A
          hU : IsOpen U
          x : A
          ⊢ Membership.mem (Set.iUnion fun k => Set.iUnion fun h => Set.preimage (fun x  …
        -/
      · rintro ⟨_, ⟨k, rfl⟩, _, ⟨(hk : φ k = 1), rfl⟩, hx⟩
        /-
          case h.mpr.intro.intro.intro.intro.intro.intro
          A : Type u_1
          inst✝⁶ : Group A
          inst✝⁵ : TopologicalSpace A
          inst✝⁴ : TopologicalGroup A
          B : Type u_2
          inst✝³ : Group B
          inst✝² : TopologicalSpace B
          F : Type u_3
          inst✝¹ : FunLike F A B
          inst✝ : MonoidHomClass F A B
          φ : F
          hφ : Topology.IsQuotientMap ⇑φ
          U : Set A
          hU : IsOpen U
          x k : A
          hk : Eq (φ k) 1
          hx : Membership.mem ((fun h => Set.preimage (fun x => HMul.hMul x k) U) hk) x
          ⊢ Membership.mem (Set.preimage (⇑φ) (Set.image (⇑φ) U)) x
        -/
        use x * k, hx
        /-
          case right
          A : Type u_1
          inst✝⁶ : Group A
          inst✝⁵ : TopologicalSpace A
          inst✝⁴ : TopologicalGroup A
          B : Type u_2
          inst✝³ : Group B
          inst✝² : TopologicalSpace B
          F : Type u_3
          inst✝¹ : FunLike F A B
          inst✝ : MonoidHomClass F A B
          φ : F
          hφ : Topology.IsQuotientMap ⇑φ
          U : Set A
          hU : IsOpen U
          x k : A
          hk : Eq (φ k) 1
          hx : Membership.mem ((fun h => Set.preimage (fun x => HMul.hMul x k) U) hk) x
          ⊢ Eq (φ (HMul.hMul x k)) (φ x)
        -/
        rw [map_mul, hk, mul_one]
        /-
          🎉 no goals
        -/


@[to_additive]
theorem TopologicalGroup.ext {G : Type*} [Group G] {t t' : TopologicalSpace G}
    (tg : @TopologicalGroup G t _) (tg' : @TopologicalGroup G t' _)
    (h : @nhds G t 1 = @nhds G t' 1) : t = t' :=
  TopologicalSpace.ext_nhds fun x ↦ by
    /-
      G : Type u_1
      inst✝ : Group G
      t t' : TopologicalSpace G
      tg : TopologicalGroup G
      tg' : TopologicalGroup G
      h : Eq (nhds 1) (nhds 1)
      x : G
      ⊢ Eq (nhds x) (nhds x)
    -/
    rw [← @nhds_translation_mul_inv G t _ _ x, ← @nhds_translation_mul_inv G t' _ _ x, ← h]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem TopologicalGroup.ext_iff {G : Type*} [Group G] {t t' : TopologicalSpace G}
    (tg : @TopologicalGroup G t _) (tg' : @TopologicalGroup G t' _) :
    t = t' ↔ @nhds G t 1 = @nhds G t' 1 :=
  ⟨fun h => h ▸ rfl, tg.ext tg'⟩


@[to_additive]
theorem ContinuousInv.of_nhds_one {G : Type*} [Group G] [TopologicalSpace G]
    (hinv : Tendsto (fun x : G => x⁻¹) (𝓝 1) (𝓝 1))
    (hleft : ∀ x₀ : G, 𝓝 x₀ = map (fun x : G => x₀ * x) (𝓝 1))
    (hconj : ∀ x₀ : G, Tendsto (fun x : G => x₀ * x * x₀⁻¹) (𝓝 1) (𝓝 1)) : ContinuousInv G := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    hinv : Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
    hleft : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
    hconj : ∀ (x₀ : G), Filter.Tendsto (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.i …
    ⊢ ContinuousInv G
  -/
  refine ⟨continuous_iff_continuousAt.2 fun x₀ => ?_⟩
  have : Tendsto (fun x => x₀⁻¹ * (x₀ * x⁻¹ * x₀⁻¹)) (𝓝 1) (map (x₀⁻¹ * ·) (𝓝 1)) :=
    (tendsto_map.comp <| hconj x₀).comp hinv
  simpa only [ContinuousAt, hleft x₀, hleft x₀⁻¹, tendsto_map'_iff, Function.comp_def, mul_assoc,
    mul_inv_rev, inv_mul_cancel_left] using this


@[to_additive]
theorem TopologicalGroup.of_nhds_one' {G : Type u} [Group G] [TopologicalSpace G]
    (hmul : Tendsto (uncurry ((· * ·) : G → G → G)) (𝓝 1 ×ˢ 𝓝 1) (𝓝 1))
    (hinv : Tendsto (fun x : G => x⁻¹) (𝓝 1) (𝓝 1))
    (hleft : ∀ x₀ : G, 𝓝 x₀ = map (fun x => x₀ * x) (𝓝 1))
    (hright : ∀ x₀ : G, 𝓝 x₀ = map (fun x => x * x₀) (𝓝 1)) : TopologicalGroup G :=
  { toContinuousMul := ContinuousMul.of_nhds_one hmul hleft hright
    toContinuousInv :=
      ContinuousInv.of_nhds_one hinv hleft fun x₀ =>
        le_of_eq
          (by
            rw [show (fun x => x₀ * x * x₀⁻¹) = (fun x => x * x₀⁻¹) ∘ fun x => x₀ * x from rfl, ←
              map_map, ← hleft, hright, map_map]
            /-
              G : Type u
              inst✝¹ : Group G
              inst✝ : TopologicalSpace G
              hmul : Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.s …
              hinv : Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
              hleft : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
              hright : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x x₀) (nhds  …
              x₀ : G
              ⊢ Eq (Filter.map (Function.comp (fun x => HMul.hMul x (Inv.inv x₀)) fun x => H …
            -/
            simp [(· ∘ ·)]) }
            /-
              🎉 no goals
            -/


@[to_additive]
theorem TopologicalGroup.of_nhds_one {G : Type u} [Group G] [TopologicalSpace G]
    (hmul : Tendsto (uncurry ((· * ·) : G → G → G)) (𝓝 1 ×ˢ 𝓝 1) (𝓝 1))
    (hinv : Tendsto (fun x : G => x⁻¹) (𝓝 1) (𝓝 1))
    (hleft : ∀ x₀ : G, 𝓝 x₀ = map (x₀ * ·) (𝓝 1))
    (hconj : ∀ x₀ : G, Tendsto (x₀ * · * x₀⁻¹) (𝓝 1) (𝓝 1)) : TopologicalGroup G := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    hmul : Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.s …
    hinv : Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
    hleft : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
    hconj : ∀ (x₀ : G), Filter.Tendsto (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.i …
    ⊢ TopologicalGroup G
  -/
  refine TopologicalGroup.of_nhds_one' hmul hinv hleft fun x₀ => ?_
  replace hconj : ∀ x₀ : G, map (x₀ * · * x₀⁻¹) (𝓝 1) = 𝓝 1 :=
    fun x₀ => map_eq_of_inverse (x₀⁻¹ * · * x₀⁻¹⁻¹) (by ext; simp [mul_assoc]) (hconj _) (hconj _)
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    hmul : Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.s …
    hinv : Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
    hleft : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
    x₀ : G
    hconj : ∀ (x₀ : G), Eq (Filter.map (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.i …
    ⊢ Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x x₀) (nhds 1))
  -/
  rw [← hconj x₀]
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : TopologicalSpace G
    hmul : Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.s …
    hinv : Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
    hleft : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
    x₀ : G
    hconj : ∀ (x₀ : G), Eq (Filter.map (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.i …
    ⊢ Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x x₀) (Filter.map (fun x => HMu …
  -/
  simpa [Function.comp_def] using hleft _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem TopologicalGroup.of_comm_of_nhds_one {G : Type u} [CommGroup G] [TopologicalSpace G]
    (hmul : Tendsto (uncurry ((· * ·) : G → G → G)) (𝓝 1 ×ˢ 𝓝 1) (𝓝 1))
    (hinv : Tendsto (fun x : G => x⁻¹) (𝓝 1) (𝓝 1))
    (hleft : ∀ x₀ : G, 𝓝 x₀ = map (x₀ * ·) (𝓝 1)) : TopologicalGroup G :=
                                                   /-
                                                     G : Type u
                                                     inst✝¹ : CommGroup G
                                                     inst✝ : TopologicalSpace G
                                                     hmul : Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.s …
                                                     hinv : Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
                                                     hleft : ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
                                                     ⊢ ∀ (x₀ : G), Filter.Tendsto (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.inv x₀) …
                                                   -/
  TopologicalGroup.of_nhds_one hmul hinv hleft (by simpa using tendsto_id)
                                                   /-
                                                     🎉 no goals
                                                   -/


variable (G) in
/-- Any first countable topological group has an antitone neighborhood basis `u : ℕ → Set G` for
which `(u (n + 1)) ^ 2 ⊆ u n`. The existence of such a neighborhood basis is a key tool for
`QuotientGroup.completeSpace` -/
@[to_additive
  "Any first countable topological additive group has an antitone neighborhood basis
  `u : ℕ → set G` for which `u (n + 1) + u (n + 1) ⊆ u n`.
  The existence of such a neighborhood basis is a key tool for `QuotientAddGroup.completeSpace`"]
theorem TopologicalGroup.exists_antitone_basis_nhds_one [FirstCountableTopology G] :
    ∃ u : ℕ → Set G, (𝓝 1).HasAntitoneBasis u ∧ ∀ n, u (n + 1) * u (n + 1) ⊆ u n := by
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : FirstCountableTopology G
    ⊢ Exists fun u => And ((nhds 1).HasAntitoneBasis u) (∀ (n : Nat), HasSubset.Su …
  -/
  rcases (𝓝 (1 : G)).exists_antitone_basis with ⟨u, hu, u_anti⟩
  have :=
    ((hu.prod_nhds hu).tendsto_iff hu).mp
      (by simpa only [mul_one] using continuous_mul.tendsto ((1, 1) : G × G))
  simp only [and_self_iff, mem_prod, and_imp, Prod.forall, exists_true_left, Prod.exists,
    forall_true_left] at this
  have event_mul : ∀ n : ℕ, ∀ᶠ m in atTop, u m * u m ⊆ u n := by
    intro n
    rcases this n with ⟨j, k, -, h⟩
    refine atTop_basis.eventually_iff.mpr ⟨max j k, True.intro, fun m hm => ?_⟩
    rintro - ⟨a, ha, b, hb, rfl⟩
    exact h a b (u_anti ((le_max_left _ _).trans hm) ha) (u_anti ((le_max_right _ _).trans hm) hb)
  /-
    case intro.mk
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : FirstCountableTopology G
    u : Nat → Set G
    hu : (nhds 1).HasBasis (fun x => True) u
    u_anti : Antitone u
    this : ∀ (ib : Nat), Exists fun a => Exists fun b => And True (∀ (a_1 b_1 : G) …
    event_mul : ∀ (n : Nat), Filter.Eventually (fun m => HasSubset.Subset (HMul.hM …
    ⊢ Exists fun u => And ((nhds 1).HasAntitoneBasis u) (∀ (n : Nat), HasSubset.Su …
  -/
  obtain ⟨φ, -, hφ, φ_anti_basis⟩ := HasAntitoneBasis.subbasis_with_rel ⟨hu, u_anti⟩ event_mul
  /-
    case intro.mk.intro.intro.intro
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : FirstCountableTopology G
    u : Nat → Set G
    hu : (nhds 1).HasBasis (fun x => True) u
    u_anti : Antitone u
    this : ∀ (ib : Nat), Exists fun a => Exists fun b => And True (∀ (a_1 b_1 : G) …
    event_mul : ∀ (n : Nat), Filter.Eventually (fun m => HasSubset.Subset (HMul.hM …
    φ : Nat → Nat
    hφ : ∀ ⦃m n : Nat⦄, LT.lt m n → HasSubset.Subset (HMul.hMul (u (φ n)) (u (φ n) …
    φ_anti_basis : (nhds 1).HasAntitoneBasis (Function.comp u φ)
    ⊢ Exists fun u => And ((nhds 1).HasAntitoneBasis u) (∀ (n : Nat), HasSubset.Su …
  -/
  exact ⟨u ∘ φ, φ_anti_basis, fun n => hφ n.lt_succ_self⟩
  /-
    🎉 no goals
  -/


/-- A typeclass saying that `p : G × G ↦ p.1 - p.2` is a continuous function. This property
automatically holds for topological additive groups but it also holds, e.g., for `ℝ≥0`. -/
class ContinuousSub (G : Type*) [TopologicalSpace G] [Sub G] : Prop where
  continuous_sub : Continuous fun p : G × G => p.1 - p.2


/-- A typeclass saying that `p : G × G ↦ p.1 / p.2` is a continuous function. This property
automatically holds for topological groups. Lemmas using this class have primes.
The unprimed version is for `GroupWithZero`. -/
@[to_additive existing]
class ContinuousDiv (G : Type*) [TopologicalSpace G] [Div G] : Prop where
  continuous_div' : Continuous fun p : G × G => p.1 / p.2

-- see Note [lower instance priority]

@[to_additive]
instance (priority := 100) TopologicalGroup.to_continuousDiv [TopologicalSpace G] [Group G]
    [TopologicalGroup G] : ContinuousDiv G :=
  ⟨by
    /-
      G : Type w
      H : Type x
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      ⊢ Continuous fun p => HDiv.hDiv p.1 p.2
    -/
    simp only [div_eq_mul_inv]
    /-
      G : Type w
      H : Type x
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      ⊢ Continuous fun p => HMul.hMul p.1 (Inv.inv p.2)
    -/
    exact continuous_fst.mul continuous_snd.inv⟩
    /-
      🎉 no goals
    -/


@[to_additive sub]
theorem Filter.Tendsto.div' {f g : α → G} {l : Filter α} {a b : G} (hf : Tendsto f l (𝓝 a))
    (hg : Tendsto g l (𝓝 b)) : Tendsto (fun x => f x / g x) l (𝓝 (a / b)) :=
  (continuous_div'.tendsto (a, b)).comp (hf.prod_mk_nhds hg)


@[to_additive const_sub]
theorem Filter.Tendsto.const_div' (b : G) {c : G} {f : α → G} {l : Filter α}
    (h : Tendsto f l (𝓝 c)) : Tendsto (fun k : α => b / f k) l (𝓝 (b / c)) :=
  tendsto_const_nhds.div' h


@[to_additive]
lemma Filter.tendsto_const_div_iff {G : Type*} [CommGroup G] [TopologicalSpace G] [ContinuousDiv G]
    (b : G) {c : G} {f : α → G} {l : Filter α} :
    Tendsto (fun k : α ↦ b / f k) l (𝓝 (b / c)) ↔ Tendsto f l (𝓝 c) := by
  /-
    α : Type u
    G : Type u_1
    inst✝² : CommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousDiv G
    b c : G
    f : α → G
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun k => HDiv.hDiv b (f k)) l (nhds (HDiv.hDiv b c))) ( …
  -/
  refine ⟨fun h ↦ ?_, Filter.Tendsto.const_div' b⟩
  /-
    α : Type u
    G : Type u_1
    inst✝² : CommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousDiv G
    b c : G
    f : α → G
    l : Filter α
    h : Filter.Tendsto (fun k => HDiv.hDiv b (f k)) l (nhds (HDiv.hDiv b c))
    ⊢ Filter.Tendsto f l (nhds c)
  -/
                                    /-
                                      🎉 no goals
                                    -/
  convert h.const_div' b with k <;> rw [div_div_cancel]
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive sub_const]
theorem Filter.Tendsto.div_const' {c : G} {f : α → G} {l : Filter α} (h : Tendsto f l (𝓝 c))
    (b : G) : Tendsto (f · / b) l (𝓝 (c / b)) :=
  h.div' tendsto_const_nhds


lemma Filter.tendsto_div_const_iff {G : Type*}
    [CommGroupWithZero G] [TopologicalSpace G] [ContinuousDiv G]
    {b : G} (hb : b ≠ 0) {c : G} {f : α → G} {l : Filter α} :
    Tendsto (f · / b) l (𝓝 (c / b)) ↔ Tendsto f l (𝓝 c) := by
  /-
    α : Type u
    G : Type u_1
    inst✝² : CommGroupWithZero G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousDiv G
    b : G
    hb : Ne b 0
    c : G
    f : α → G
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) b) l (nhds (HDiv.hDiv c b))) ( …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ Filter.Tendsto.div_const' h b⟩
  /-
    α : Type u
    G : Type u_1
    inst✝² : CommGroupWithZero G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousDiv G
    b : G
    hb : Ne b 0
    c : G
    f : α → G
    l : Filter α
    h : Filter.Tendsto (fun x => HDiv.hDiv (f x) b) l (nhds (HDiv.hDiv c b))
    ⊢ Filter.Tendsto f l (nhds c)
  -/
                                      /-
                                        🎉 no goals
                                      -/
  convert h.div_const' b⁻¹ with k <;> rw [div_div, mul_inv_cancel₀ hb, div_one]
                                      /-
                                        🎉 no goals
                                      -/


lemma Filter.tendsto_sub_const_iff {G : Type*}
    [AddCommGroup G] [TopologicalSpace G] [ContinuousSub G]
    (b : G) {c : G} {f : α → G} {l : Filter α} :
    Tendsto (f · - b) l (𝓝 (c - b)) ↔ Tendsto f l (𝓝 c) := by
  /-
    α : Type u
    G : Type u_1
    inst✝² : AddCommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousSub G
    b c : G
    f : α → G
    l : Filter α
    ⊢ Iff (Filter.Tendsto (fun x => HSub.hSub (f x) b) l (nhds (HSub.hSub c b))) ( …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ Filter.Tendsto.sub_const h b⟩
  /-
    α : Type u
    G : Type u_1
    inst✝² : AddCommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousSub G
    b c : G
    f : α → G
    l : Filter α
    h : Filter.Tendsto (fun x => HSub.hSub (f x) b) l (nhds (HSub.hSub c b))
    ⊢ Filter.Tendsto f l (nhds c)
  -/
                                      /-
                                        🎉 no goals
                                      -/
  convert h.sub_const (-b) with k <;> rw [sub_sub, ← sub_eq_add_neg, sub_self, sub_zero]
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive (attr := continuity, fun_prop) sub]
theorem Continuous.div' (hf : Continuous f) (hg : Continuous g) : Continuous fun x => f x / g x :=
  continuous_div'.comp (hf.prod_mk hg : _)


@[to_additive (attr := continuity) continuous_sub_left]
lemma continuous_div_left' (a : G) : Continuous (a / ·) := continuous_const.div' continuous_id


@[to_additive (attr := continuity) continuous_sub_right]
lemma continuous_div_right' (a : G) : Continuous (· / a) := continuous_id.div' continuous_const


@[to_additive (attr := fun_prop) sub]
theorem ContinuousAt.div' {f g : α → G} {x : α} (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun x => f x / g x) x :=
  Filter.Tendsto.div' hf hg


@[to_additive sub]
theorem ContinuousWithinAt.div' (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (fun x => f x / g x) s x :=
  Filter.Tendsto.div' hf hg


@[to_additive (attr := fun_prop) sub]
theorem ContinuousOn.div' (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun x => f x / g x) s := fun x hx => (hf x hx).div' (hg x hx)


/-- A version of `Homeomorph.mulLeft a b⁻¹` that is defeq to `a / b`. -/
@[to_additive (attr := simps! (config := { simpRhs := true }))
  " A version of `Homeomorph.addLeft a (-b)` that is defeq to `a - b`. "]
def Homeomorph.divLeft (x : G) : G ≃ₜ G :=
  { Equiv.divLeft x with
    continuous_toFun := continuous_const.div' continuous_id
    continuous_invFun := continuous_inv.mul continuous_const }


@[to_additive]
theorem isOpenMap_div_left (a : G) : IsOpenMap (a / ·) :=
  (Homeomorph.divLeft _).isOpenMap


@[to_additive]
theorem isClosedMap_div_left (a : G) : IsClosedMap (a / ·) :=
  (Homeomorph.divLeft _).isClosedMap


/-- A version of `Homeomorph.mulRight a⁻¹ b` that is defeq to `b / a`. -/
@[to_additive (attr := simps! (config := { simpRhs := true }))
  "A version of `Homeomorph.addRight (-a) b` that is defeq to `b - a`. "]
def Homeomorph.divRight (x : G) : G ≃ₜ G :=
  { Equiv.divRight x with
    continuous_toFun := continuous_id.div' continuous_const
    continuous_invFun := continuous_id.mul continuous_const }


@[to_additive]
lemma isOpenMap_div_right (a : G) : IsOpenMap (· / a) := (Homeomorph.divRight a).isOpenMap


@[to_additive]
lemma isClosedMap_div_right (a : G) : IsClosedMap (· / a) := (Homeomorph.divRight a).isClosedMap


@[to_additive]
theorem tendsto_div_nhds_one_iff {α : Type*} {l : Filter α} {x : G} {u : α → G} :
    Tendsto (u · / x) l (𝓝 1) ↔ Tendsto u l (𝓝 x) :=
  haveI A : Tendsto (fun _ : α => x) l (𝓝 x) := tendsto_const_nhds
               /-
                 G : Type w
                 inst✝² : Group G
                 inst✝¹ : TopologicalSpace G
                 inst✝ : TopologicalGroup G
                 α : Type u_1
                 l : Filter α
                 x : G
                 u : α → G
                 A : Filter.Tendsto (fun x_1 => x) l (nhds x)
                 h : Filter.Tendsto (fun x_1 => HDiv.hDiv (u x_1) x) l (nhds 1)
                 ⊢ Filter.Tendsto u l (nhds x)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simpa using h.mul A, fun h => by simpa using h.div' A⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
theorem nhds_translation_div (x : G) : comap (· / x) (𝓝 1) = 𝓝 x := by
  /-
    G : Type w
    inst✝² : Group G
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalGroup G
    x : G
    ⊢ Eq (Filter.comap (fun x_1 => HDiv.hDiv x_1 x) (nhds 1)) (nhds x)
  -/
  simpa only [div_eq_mul_inv] using nhds_translation_mul_inv x
  /-
    🎉 no goals
  -/


@[to_additive]
theorem subset_interior_smul : interior s • interior t ⊆ interior (s • t) :=
  (Set.smul_subset_smul_right interior_subset).trans subset_interior_smul_right


@[to_additive]
theorem IsClosed.smul_left_of_isCompact (ht : IsClosed t) (hs : IsCompact s) :
    IsClosed (s • t) := by
  have : ∀ x ∈ s • t, ∃ g ∈ s, g⁻¹ • x ∈ t := by
    rintro x ⟨g, hgs, y, hyt, rfl⟩
    refine ⟨g, hgs, ?_⟩
    rwa [inv_smul_smul]
  /-
    α : Type u
    β : Type v
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Group α
    inst✝² : MulAction α β
    inst✝¹ : ContinuousInv α
    inst✝ : ContinuousSMul α β
    s : Set α
    t : Set β
    ht : IsClosed t
    hs : IsCompact s
    this : ∀ (x : β), Membership.mem (HSMul.hSMul s t) x → Exists fun g => And (Me …
    ⊢ IsClosed (HSMul.hSMul s t)
  -/
  choose! f hf using this
  /-
    α : Type u
    β : Type v
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Group α
    inst✝² : MulAction α β
    inst✝¹ : ContinuousInv α
    inst✝ : ContinuousSMul α β
    s : Set α
    t : Set β
    ht : IsClosed t
    hs : IsCompact s
    f : β → α
    hf : ∀ (x : β), Membership.mem (HSMul.hSMul s t) x → And (Membership.mem s (f  …
    ⊢ IsClosed (HSMul.hSMul s t)
  -/
  refine isClosed_of_closure_subset (fun x hx ↦ ?_)
  /-
    α : Type u
    β : Type v
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Group α
    inst✝² : MulAction α β
    inst✝¹ : ContinuousInv α
    inst✝ : ContinuousSMul α β
    s : Set α
    t : Set β
    ht : IsClosed t
    hs : IsCompact s
    f : β → α
    hf : ∀ (x : β), Membership.mem (HSMul.hSMul s t) x → And (Membership.mem s (f  …
    x : β
    hx : Membership.mem (closure (HSMul.hSMul s t)) x
    ⊢ Membership.mem (HSMul.hSMul s t) x
  -/
  rcases mem_closure_iff_ultrafilter.mp hx with ⟨u, hust, hux⟩
  have : Ultrafilter.map f u ≤ 𝓟 s :=
    calc Ultrafilter.map f u ≤ map f (𝓟 (s • t)) := map_mono (le_principal_iff.mpr hust)
      _ = 𝓟 (f '' (s • t)) := map_principal
      _ ≤ 𝓟 s := principal_mono.mpr (image_subset_iff.mpr (fun x hx ↦ (hf x hx).1))
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Group α
    inst✝² : MulAction α β
    inst✝¹ : ContinuousInv α
    inst✝ : ContinuousSMul α β
    s : Set α
    t : Set β
    ht : IsClosed t
    hs : IsCompact s
    f : β → α
    hf : ∀ (x : β), Membership.mem (HSMul.hSMul s t) x → And (Membership.mem s (f  …
    x : β
    hx : Membership.mem (closure (HSMul.hSMul s t)) x
    u : Ultrafilter β
    hust : Membership.mem u (HSMul.hSMul s t)
    hux : LE.le (↑u) (nhds x)
    this : LE.le (↑(Ultrafilter.map f u)) (Filter.principal s)
    ⊢ Membership.mem (HSMul.hSMul s t) x
  -/
  rcases hs.ultrafilter_le_nhds (Ultrafilter.map f u) this with ⟨g, hg, hug⟩
  suffices g⁻¹ • x ∈ t from
    ⟨g, hg, g⁻¹ • x, this, smul_inv_smul _ _⟩
  exact ht.mem_of_tendsto ((Tendsto.inv hug).smul hux)
    (Eventually.mono hust (fun y hy ↦ (hf y hy).2))


@[to_additive]
theorem MulAction.isClosedMap_quotient [CompactSpace α] :
    letI := orbitRel α β
    IsClosedMap (Quotient.mk' : β → Quotient (orbitRel α β)) := by
  /-
    α : Type u
    β : Type v
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Group α
    inst✝³ : MulAction α β
    inst✝² : ContinuousInv α
    inst✝¹ : ContinuousSMul α β
    inst✝ : CompactSpace α
    ⊢ IsClosedMap Quotient.mk'
  -/
  intro t ht
  rw [← isQuotientMap_quotient_mk'.isClosed_preimage,
    MulAction.quotient_preimage_image_eq_union_mul]
  /-
    α : Type u
    β : Type v
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Group α
    inst✝³ : MulAction α β
    inst✝² : ContinuousInv α
    inst✝¹ : ContinuousSMul α β
    inst✝ : CompactSpace α
    t : Set β
    ht : IsClosed t
    ⊢ IsClosed (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x) t)
  -/
  convert ht.smul_left_of_isCompact (isCompact_univ (X := α))
  /-
    case h.e'_3
    α : Type u
    β : Type v
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Group α
    inst✝³ : MulAction α β
    inst✝² : ContinuousInv α
    inst✝¹ : ContinuousSMul α β
    inst✝ : CompactSpace α
    t : Set β
    ht : IsClosed t
    ⊢ Eq (Set.iUnion fun g => Set.image (fun x => HSMul.hSMul g x) t) (HSMul.hSMul …
  -/
  rw [← biUnion_univ, ← iUnion_smul_left_image]
  /-
    case h.e'_3
    α : Type u
    β : Type v
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : Group α
    inst✝³ : MulAction α β
    inst✝² : ContinuousInv α
    inst✝¹ : ContinuousSMul α β
    inst✝ : CompactSpace α
    t : Set β
    ht : IsClosed t
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => Set.image (fun x_1 => HSMul.hSMu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOpen.mul_left : IsOpen t → IsOpen (s * t) :=
  IsOpen.smul_left


@[to_additive]
theorem subset_interior_mul_right : s * interior t ⊆ interior (s * t) :=
  subset_interior_smul_right


@[to_additive]
theorem subset_interior_mul : interior s * interior t ⊆ interior (s * t) :=
  subset_interior_smul


@[to_additive]
theorem singleton_mul_mem_nhds (a : α) {b : α} (h : s ∈ 𝓝 b) : {a} * s ∈ 𝓝 (a * b) := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul α α
    s : Set α
    a b : α
    h : Membership.mem (nhds b) s
    ⊢ Membership.mem (nhds (HMul.hMul a b)) (HMul.hMul (Singleton.singleton a) s)
  -/
  rwa [← smul_eq_mul, ← smul_eq_mul, singleton_smul, smul_mem_nhds_smul_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem singleton_mul_mem_nhds_of_nhds_one (a : α) (h : s ∈ 𝓝 (1 : α)) : {a} * s ∈ 𝓝 a := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul α α
    s : Set α
    a : α
    h : Membership.mem (nhds 1) s
    ⊢ Membership.mem (nhds a) (HMul.hMul (Singleton.singleton a) s)
  -/
  simpa only [mul_one] using singleton_mul_mem_nhds a h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOpen.mul_right (hs : IsOpen s) : IsOpen (s * t) := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul (MulOpposite α) α
    s t : Set α
    hs : IsOpen s
    ⊢ IsOpen (HMul.hMul s t)
  -/
  rw [← image_op_smul]
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul (MulOpposite α) α
    s t : Set α
    hs : IsOpen s
    ⊢ IsOpen (HSMul.hSMul (Set.image MulOpposite.op t) s)
  -/
  exact hs.smul_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem subset_interior_mul_left : interior s * t ⊆ interior (s * t) :=
  interior_maximal (Set.mul_subset_mul_right interior_subset) isOpen_interior.mul_right


@[to_additive]
theorem subset_interior_mul' : interior s * interior t ⊆ interior (s * t) :=
  (Set.mul_subset_mul_left interior_subset).trans subset_interior_mul_left


@[to_additive]
theorem mul_singleton_mem_nhds (a : α) {b : α} (h : s ∈ 𝓝 b) : s * {a} ∈ 𝓝 (b * a) := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul (MulOpposite α) α
    s : Set α
    a b : α
    h : Membership.mem (nhds b) s
    ⊢ Membership.mem (nhds (HMul.hMul b a)) (HMul.hMul s (Singleton.singleton a))
  -/
  rw [mul_singleton]
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul (MulOpposite α) α
    s : Set α
    a b : α
    h : Membership.mem (nhds b) s
    ⊢ Membership.mem (nhds (HMul.hMul b a)) (Set.image (fun x => HMul.hMul x a) s)
  -/
  exact smul_mem_nhds_smul (op a) h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_singleton_mem_nhds_of_nhds_one (a : α) (h : s ∈ 𝓝 (1 : α)) : s * {a} ∈ 𝓝 a := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    inst✝¹ : Group α
    inst✝ : ContinuousConstSMul (MulOpposite α) α
    s : Set α
    a : α
    h : Membership.mem (nhds 1) s
    ⊢ Membership.mem (nhds a) (HMul.hMul s (Singleton.singleton a))
  -/
  simpa only [one_mul] using mul_singleton_mem_nhds a h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOpen.div_left (ht : IsOpen t) : IsOpen (s / t) := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s t : Set G
    ht : IsOpen t
    ⊢ IsOpen (HDiv.hDiv s t)
  -/
  rw [← iUnion_div_left_image]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s t : Set G
    ht : IsOpen t
    ⊢ IsOpen (Set.iUnion fun a => Set.iUnion fun h => Set.image (fun x => HDiv.hDi …
  -/
  exact isOpen_biUnion fun a _ => isOpenMap_div_left a t ht
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOpen.div_right (hs : IsOpen s) : IsOpen (s / t) := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s t : Set G
    hs : IsOpen s
    ⊢ IsOpen (HDiv.hDiv s t)
  -/
  rw [← iUnion_div_right_image]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s t : Set G
    hs : IsOpen s
    ⊢ IsOpen (Set.iUnion fun a => Set.iUnion fun h => Set.image (fun x => HDiv.hDi …
  -/
  exact isOpen_biUnion fun a _ => isOpenMap_div_right a s hs
  /-
    🎉 no goals
  -/


@[to_additive]
theorem subset_interior_div_left : interior s / t ⊆ interior (s / t) :=
  interior_maximal (div_subset_div_right interior_subset) isOpen_interior.div_right


@[to_additive]
theorem subset_interior_div_right : s / interior t ⊆ interior (s / t) :=
  interior_maximal (div_subset_div_left interior_subset) isOpen_interior.div_left


@[to_additive]
theorem subset_interior_div : interior s / interior t ⊆ interior (s / t) :=
  (div_subset_div_left interior_subset).trans subset_interior_div_left


@[to_additive]
theorem IsOpen.mul_closure (hs : IsOpen s) (t : Set G) : s * closure t = s * t := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s : Set G
    hs : IsOpen s
    t : Set G
    ⊢ Eq (HMul.hMul s (closure t)) (HMul.hMul s t)
  -/
  refine (mul_subset_iff.2 fun a ha b hb => ?_).antisymm (mul_subset_mul_left subset_closure)
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s : Set G
    hs : IsOpen s
    t : Set G
    a : G
    ha : Membership.mem s a
    b : G
    hb : Membership.mem (closure t) b
    ⊢ Membership.mem (HMul.hMul s t) (HMul.hMul a b)
  -/
  rw [mem_closure_iff] at hb
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s : Set G
    hs : IsOpen s
    t : Set G
    a : G
    ha : Membership.mem s a
    b : G
    hb : ∀ (o : Set G), IsOpen o → Membership.mem o b → (Inter.inter o t).Nonempty
    ⊢ Membership.mem (HMul.hMul s t) (HMul.hMul a b)
  -/
  have hbU : b ∈ s⁻¹ * {a * b} := ⟨a⁻¹, Set.inv_mem_inv.2 ha, a * b, rfl, inv_mul_cancel_left _ _⟩
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s : Set G
    hs : IsOpen s
    t : Set G
    a : G
    ha : Membership.mem s a
    b : G
    hb : ∀ (o : Set G), IsOpen o → Membership.mem o b → (Inter.inter o t).Nonempty
    hbU : Membership.mem (HMul.hMul (Inv.inv s) (Singleton.singleton (HMul.hMul a  …
    ⊢ Membership.mem (HMul.hMul s t) (HMul.hMul a b)
  -/
  obtain ⟨_, ⟨c, hc, d, rfl : d = _, rfl⟩, hcs⟩ := hb _ hs.inv.mul_right hbU
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s : Set G
    hs : IsOpen s
    t : Set G
    a : G
    ha : Membership.mem s a
    b : G
    hb : ∀ (o : Set G), IsOpen o → Membership.mem o b → (Inter.inter o t).Nonempty
    hbU : Membership.mem (HMul.hMul (Inv.inv s) (Singleton.singleton (HMul.hMul a  …
    c : G
    hc : Membership.mem (Inv.inv s) c
    hcs : Membership.mem t ((fun x1 x2 => HMul.hMul x1 x2) c (HMul.hMul a b))
    ⊢ Membership.mem (HMul.hMul s t) (HMul.hMul a b)
  -/
  exact ⟨c⁻¹, hc, _, hcs, inv_mul_cancel_left _ _⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOpen.closure_mul (ht : IsOpen t) (s : Set G) : closure s * t = s * t := by
  rw [← inv_inv (closure s * t), mul_inv_rev, inv_closure, ht.inv.mul_closure, mul_inv_rev, inv_inv,
    inv_inv]


@[to_additive]
theorem IsOpen.div_closure (hs : IsOpen s) (t : Set G) : s / closure t = s / t := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s : Set G
    hs : IsOpen s
    t : Set G
    ⊢ Eq (HDiv.hDiv s (closure t)) (HDiv.hDiv s t)
  -/
  simp_rw [div_eq_mul_inv, inv_closure, hs.mul_closure]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOpen.closure_div (ht : IsOpen t) (s : Set G) : closure s / t = s / t := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    t : Set G
    ht : IsOpen t
    s : Set G
    ⊢ Eq (HDiv.hDiv (closure s) t) (HDiv.hDiv s t)
  -/
  simp_rw [div_eq_mul_inv, ht.inv.closure_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsClosed.mul_left_of_isCompact (ht : IsClosed t) (hs : IsCompact s) : IsClosed (s * t) :=
  ht.smul_left_of_isCompact hs


@[to_additive]
theorem IsClosed.mul_right_of_isCompact (ht : IsClosed t) (hs : IsCompact s) :
    IsClosed (t * s) := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s t : Set G
    ht : IsClosed t
    hs : IsCompact s
    ⊢ IsClosed (HMul.hMul t s)
  -/
  rw [← image_op_smul]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    s t : Set G
    ht : IsClosed t
    hs : IsCompact s
    ⊢ IsClosed (HSMul.hSMul (Set.image MulOpposite.op s) t)
  -/
  exact IsClosed.smul_left_of_isCompact ht (hs.image continuous_op)
  /-
    🎉 no goals
  -/


@[to_additive]
lemma subset_mul_closure_one {G} [MulOneClass G] [TopologicalSpace G] (s : Set G) :
    s ⊆ s * (closure {1} : Set G) := by
  /-
    G : Type u_1
    inst✝¹ : MulOneClass G
    inst✝ : TopologicalSpace G
    s : Set G
    ⊢ HasSubset.Subset s (HMul.hMul s (closure (Singleton.singleton 1)))
  -/
  have : s ⊆ s * ({1} : Set G) := by simp
  /-
    G : Type u_1
    inst✝¹ : MulOneClass G
    inst✝ : TopologicalSpace G
    s : Set G
    this : HasSubset.Subset s (HMul.hMul s (Singleton.singleton 1))
    ⊢ HasSubset.Subset s (HMul.hMul s (closure (Singleton.singleton 1)))
  -/
  exact this.trans (smul_subset_smul_left subset_closure)
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsCompact.mul_closure_one_eq_closure {K : Set G} (hK : IsCompact K) :
    K * (closure {1} : Set G) = closure K := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    ⊢ Eq (HMul.hMul K (closure (Singleton.singleton 1))) (closure K)
  -/
  apply Subset.antisymm ?_ ?_
  · calc
    K * (closure {1} : Set G) ⊆ closure K * (closure {1} : Set G) :=
      smul_subset_smul_right subset_closure
    _ ⊆ closure (K * ({1} : Set G)) := smul_set_closure_subset _ _
    _ = closure K := by simp
  · have : IsClosed (K * (closure {1} : Set G)) :=
      IsClosed.smul_left_of_isCompact isClosed_closure hK
    /-
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      K : Set G
      hK : IsCompact K
      this : IsClosed (HMul.hMul K (closure (Singleton.singleton 1)))
      ⊢ HasSubset.Subset (closure K) (HMul.hMul K (closure (Singleton.singleton 1)))
    -/
    rw [IsClosed.closure_subset_iff this]
    /-
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      K : Set G
      hK : IsCompact K
      this : IsClosed (HMul.hMul K (closure (Singleton.singleton 1)))
      ⊢ HasSubset.Subset K (HMul.hMul K (closure (Singleton.singleton 1)))
    -/
    exact subset_mul_closure_one K
    /-
      🎉 no goals
    -/


@[to_additive]
lemma IsClosed.mul_closure_one_eq {F : Set G} (hF : IsClosed F) :
    F * (closure {1} : Set G) = F := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    F : Set G
    hF : IsClosed F
    ⊢ Eq (HMul.hMul F (closure (Singleton.singleton 1))) F
  -/
  refine Subset.antisymm ?_ (subset_mul_closure_one F)
  calc
  F * (closure {1} : Set G) = closure F * closure ({1} : Set G) := by rw [hF.closure_eq]
  _ ⊆ closure (F * ({1} : Set G)) := smul_set_closure_subset _ _
  _ = F := by simp [hF.closure_eq]


@[to_additive]
lemma compl_mul_closure_one_eq {t : Set G} (ht : t * (closure {1} : Set G) = t) :
    tᶜ * (closure {1} : Set G) = tᶜ := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    t : Set G
    ht : Eq (HMul.hMul t (closure (Singleton.singleton 1))) t
    ⊢ Eq (HMul.hMul (HasCompl.compl t) (closure (Singleton.singleton 1))) (HasComp …
  -/
  refine Subset.antisymm ?_ (subset_mul_closure_one tᶜ)
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    t : Set G
    ht : Eq (HMul.hMul t (closure (Singleton.singleton 1))) t
    ⊢ HasSubset.Subset (HMul.hMul (HasCompl.compl t) (closure (Singleton.singleton …
  -/
  rintro - ⟨x, hx, g, hg, rfl⟩
  /-
    case intro.intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    t : Set G
    ht : Eq (HMul.hMul t (closure (Singleton.singleton 1))) t
    x : G
    hx : Membership.mem (HasCompl.compl t) x
    g : G
    hg : Membership.mem (closure (Singleton.singleton 1)) g
    ⊢ Membership.mem (HasCompl.compl t) ((fun x1 x2 => HMul.hMul x1 x2) x g)
  -/
  by_contra H
  have : x ∈ t * (closure {1} : Set G) := by
    rw [← Subgroup.coe_topologicalClosure_bot G] at hg ⊢
    simp only [smul_eq_mul, mem_compl_iff, not_not] at H
    exact ⟨x * g, H, g⁻¹, Subgroup.inv_mem _ hg, by simp⟩
  /-
    case intro.intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    t : Set G
    ht : Eq (HMul.hMul t (closure (Singleton.singleton 1))) t
    x : G
    hx : Membership.mem (HasCompl.compl t) x
    g : G
    hg : Membership.mem (closure (Singleton.singleton 1)) g
    H : Not (Membership.mem (HasCompl.compl t) ((fun x1 x2 => HMul.hMul x1 x2) x g))
    this : Membership.mem (HMul.hMul t (closure (Singleton.singleton 1))) x
    ⊢ False
  -/
  rw [ht] at this
  /-
    case intro.intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    t : Set G
    ht : Eq (HMul.hMul t (closure (Singleton.singleton 1))) t
    x : G
    hx : Membership.mem (HasCompl.compl t) x
    g : G
    hg : Membership.mem (closure (Singleton.singleton 1)) g
    H : Not (Membership.mem (HasCompl.compl t) ((fun x1 x2 => HMul.hMul x1 x2) x g))
    this : Membership.mem t x
    ⊢ False
  -/
  exact hx this
  /-
    🎉 no goals
  -/


@[to_additive]
lemma compl_mul_closure_one_eq_iff {t : Set G} :
    tᶜ * (closure {1} : Set G) = tᶜ ↔ t * (closure {1} : Set G) = t :=
              /-
                G : Type w
                inst✝² : TopologicalSpace G
                inst✝¹ : Group G
                inst✝ : TopologicalGroup G
                t : Set G
                h : Eq (HMul.hMul (HasCompl.compl t) (closure (Singleton.singleton 1))) (HasCo …
                ⊢ Eq (HMul.hMul t (closure (Singleton.singleton 1))) t
              -/
  ⟨fun h ↦ by simpa using compl_mul_closure_one_eq h, fun h ↦ compl_mul_closure_one_eq h⟩
              /-
                🎉 no goals
              -/


@[to_additive]
lemma IsOpen.mul_closure_one_eq {U : Set G} (hU : IsOpen U) :
    U * (closure {1} : Set G) = U :=
  compl_mul_closure_one_eq_iff.1 (hU.isClosed_compl.mul_closure_one_eq)


@[to_additive]
theorem TopologicalGroup.t1Space (h : @IsClosed G _ {1}) : T1Space G :=
               /-
                 G : Type w
                 inst✝² : TopologicalSpace G
                 inst✝¹ : Group G
                 inst✝ : ContinuousMul G
                 h : IsClosed (Singleton.singleton 1)
                 x : G
                 ⊢ IsClosed (Singleton.singleton x)
               -/
  ⟨fun x => by simpa using isClosedMap_mul_right x _ h⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
instance (priority := 100) TopologicalGroup.regularSpace : RegularSpace G := by
  /-
    G : Type w
    H : Type x
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    ⊢ RegularSpace G
  -/
  refine .of_exists_mem_nhds_isClosed_subset fun a s hs ↦ ?_
  have : Tendsto (fun p : G × G => p.1 * p.2) (𝓝 (a, 1)) (𝓝 a) :=
    continuous_mul.tendsto' _ _ (mul_one a)
  /-
    G : Type w
    H : Type x
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    a : G
    s : Set G
    hs : Membership.mem (nhds a) s
    this : Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := a, snd := 1  …
    ⊢ Exists fun t => And (Membership.mem (nhds a) t) (And (IsClosed t) (HasSubset …
  -/
  rcases mem_nhds_prod_iff.mp (this hs) with ⟨U, hU, V, hV, hUV⟩
  /-
    case intro.intro.intro.intro
    G : Type w
    H : Type x
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    a : G
    s : Set G
    hs : Membership.mem (nhds a) s
    this : Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := a, snd := 1  …
    U : Set G
    hU : Membership.mem (nhds a) U
    V : Set G
    hV : Membership.mem (nhds 1) V
    hUV : HasSubset.Subset (SProd.sprod U V) (Set.preimage (fun p => HMul.hMul p.1 …
    ⊢ Exists fun t => And (Membership.mem (nhds a) t) (And (IsClosed t) (HasSubset …
  -/
  rw [← image_subset_iff, image_prod] at hUV
  /-
    case intro.intro.intro.intro
    G : Type w
    H : Type x
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    a : G
    s : Set G
    hs : Membership.mem (nhds a) s
    this : Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (nhds { fst := a, snd := 1  …
    U : Set G
    hU : Membership.mem (nhds a) U
    V : Set G
    hV : Membership.mem (nhds 1) V
    hUV : HasSubset.Subset (Set.image2 HMul.hMul U V) s
    ⊢ Exists fun t => And (Membership.mem (nhds a) t) (And (IsClosed t) (HasSubset …
  -/
  refine ⟨closure U, mem_of_superset hU subset_closure, isClosed_closure, ?_⟩
  calc
    closure U ⊆ closure U * interior V := subset_mul_left _ (mem_interior_iff_mem_nhds.2 hV)
    _ = U * interior V := isOpen_interior.closure_mul U
    _ ⊆ U * V := mul_subset_mul_left interior_subset
    _ ⊆ s := hUV

-- `inferInstance` can find these instances now


@[to_additive]
theorem group_inseparable_iff {x y : G} : Inseparable x y ↔ x / y ∈ closure (1 : Set G) := by
  rw [← singleton_one, ← specializes_iff_mem_closure, specializes_comm, specializes_iff_inseparable,
    ← (Homeomorph.mulRight y⁻¹).isEmbedding.inseparable_iff]
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    x y : G
    ⊢ Iff (Inseparable ((Homeomorph.mulRight (Inv.inv y)) x) ((Homeomorph.mulRight …
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem TopologicalGroup.t2Space_iff_one_closed : T2Space G ↔ IsClosed ({1} : Set G) :=
  ⟨fun _ ↦ isClosed_singleton, fun h ↦
    have := TopologicalGroup.t1Space G h; inferInstance⟩


@[to_additive]
theorem TopologicalGroup.t2Space_of_one_sep (H : ∀ x : G, x ≠ 1 → ∃ U ∈ 𝓝 (1 : G), x ∉ U) :
    T2Space G := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : ∀ (x : G), Ne x 1 → Exists fun U => And (Membership.mem (nhds 1) U) (Not ( …
    ⊢ T2Space G
  -/
  suffices T1Space G from inferInstance
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : ∀ (x : G), Ne x 1 → Exists fun U => And (Membership.mem (nhds 1) U) (Not ( …
    ⊢ T1Space G
  -/
  refine t1Space_iff_specializes_imp_eq.2 fun x y hspec ↦ by_contra fun hne ↦ ?_
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : ∀ (x : G), Ne x 1 → Exists fun U => And (Membership.mem (nhds 1) U) (Not ( …
    x y : G
    hspec : Specializes x y
    hne : Not (Eq x y)
    ⊢ False
  -/
  rcases H (x * y⁻¹) (by rwa [Ne, mul_inv_eq_one]) with ⟨U, hU₁, hU⟩
  /-
    case intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    H : ∀ (x : G), Ne x 1 → Exists fun U => And (Membership.mem (nhds 1) U) (Not ( …
    x y : G
    hspec : Specializes x y
    hne : Not (Eq x y)
    U : Set G
    hU₁ : Membership.mem (nhds 1) U
    hU : Not (Membership.mem U (HMul.hMul x (Inv.inv y)))
    ⊢ False
  -/
  exact hU <| mem_of_mem_nhds <| hspec.map (continuous_mul_right y⁻¹) (by rwa [mul_inv_cancel])
  /-
    🎉 no goals
  -/


/-- Given a neighborhood `U` of the identity, one may find a neighborhood `V` of the identity which
is closed, symmetric, and satisfies `V * V ⊆ U`. -/
@[to_additive "Given a neighborhood `U` of the identity, one may find a neighborhood `V` of the
identity which is closed, symmetric, and satisfies `V + V ⊆ U`."]
theorem exists_closed_nhds_one_inv_eq_mul_subset {U : Set G} (hU : U ∈ 𝓝 1) :
    ∃ V ∈ 𝓝 1, IsClosed V ∧ V⁻¹ = V ∧ V * V ⊆ U := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    U : Set G
    hU : Membership.mem (nhds 1) U
    ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (And (IsClosed V) (And (Eq ( …
  -/
  rcases exists_open_nhds_one_mul_subset hU with ⟨V, V_open, V_mem, hV⟩
  /-
    case intro.intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : Set G
    V_open : IsOpen V
    V_mem : Membership.mem V 1
    hV : HasSubset.Subset (HMul.hMul V V) U
    ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (And (IsClosed V) (And (Eq ( …
  -/
  rcases exists_mem_nhds_isClosed_subset (V_open.mem_nhds V_mem) with ⟨W, W_mem, W_closed, hW⟩
  refine ⟨W ∩ W⁻¹, Filter.inter_mem W_mem (inv_mem_nhds_one G W_mem), W_closed.inter W_closed.inv,
    by simp [inter_comm], ?_⟩
  calc
  W ∩ W⁻¹ * (W ∩ W⁻¹)
    ⊆ W * W := mul_subset_mul inter_subset_left inter_subset_left
  _ ⊆ V * V := mul_subset_mul hW hW
  _ ⊆ U := hV


/-- A subgroup `S` of a topological group `G` acts on `G` properly discontinuously on the left, if
it is discrete in the sense that `S ∩ K` is finite for all compact `K`. (See also
`DiscreteTopology`.) -/
@[to_additive
  "A subgroup `S` of an additive topological group `G` acts on `G` properly
  discontinuously on the left, if it is discrete in the sense that `S ∩ K` is finite for all compact
  `K`. (See also `DiscreteTopology`."]
theorem Subgroup.properlyDiscontinuousSMul_of_tendsto_cofinite (S : Subgroup G)
    (hS : Tendsto S.subtype cofinite (cocompact G)) : ProperlyDiscontinuousSMul S G :=
  { finite_disjoint_inter_image := by
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        ⊢ ∀ {K L : Set G}, IsCompact K → IsCompact L → (setOf fun γ => Ne (Inter.inter …
      -/
      intro K L hK hL
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        ⊢ (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K) L)  …
      -/
      have H : Set.Finite _ := hS ((hL.prod hK).image continuous_div').compl_mem_cocompact
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        H : (HasCompl.compl (Set.preimage (⇑S.subtype) (HasCompl.compl (Set.image (fun …
        ⊢ (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K) L)  …
      -/
      rw [preimage_compl, compl_compl] at H
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        H : (Set.preimage (⇑S.subtype) (Set.image (fun p => HDiv.hDiv p.1 p.2) (SProd. …
        ⊢ (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K) L)  …
      -/
      convert H
      /-
        case h.e'_2
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        H : (Set.preimage (⇑S.subtype) (Set.image (fun p => HDiv.hDiv p.1 p.2) (SProd. …
        ⊢ Eq (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K)  …
      -/
      ext x
      /-
        case h.e'_2.h
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        H : (Set.preimage (⇑S.subtype) (Set.image (fun p => HDiv.hDiv p.1 p.2) (SProd. …
        x : Subtype fun x => Membership.mem S x
        ⊢ Iff (Membership.mem (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSM …
      -/
      simp only [image_smul, mem_setOf_eq, coeSubtype, mem_preimage, mem_image, Prod.exists]
      /-
        case h.e'_2.h
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        H : (Set.preimage (⇑S.subtype) (Set.image (fun p => HDiv.hDiv p.1 p.2) (SProd. …
        x : Subtype fun x => Membership.mem S x
        ⊢ Iff (Ne (Inter.inter (HSMul.hSMul x K) L) EmptyCollection.emptyCollection) ( …
      -/
      exact Set.smul_inter_ne_empty_iff' }
      /-
        🎉 no goals
      -/

-- attribute [local semireducible] MulOpposite -- Porting note: doesn't work in Lean 4


/-- A subgroup `S` of a topological group `G` acts on `G` properly discontinuously on the right, if
it is discrete in the sense that `S ∩ K` is finite for all compact `K`. (See also
`DiscreteTopology`.)

If `G` is Hausdorff, this can be combined with `t2Space_of_properlyDiscontinuousSMul_of_t2Space`
to show that the quotient group `G ⧸ S` is Hausdorff. -/
@[to_additive
  "A subgroup `S` of an additive topological group `G` acts on `G` properly discontinuously
  on the right, if it is discrete in the sense that `S ∩ K` is finite for all compact `K`.
  (See also `DiscreteTopology`.)

  If `G` is Hausdorff, this can be combined with `t2Space_of_properlyDiscontinuousVAdd_of_t2Space`
  to show that the quotient group `G ⧸ S` is Hausdorff."]
theorem Subgroup.properlyDiscontinuousSMul_opposite_of_tendsto_cofinite (S : Subgroup G)
    (hS : Tendsto S.subtype cofinite (cocompact G)) : ProperlyDiscontinuousSMul S.op G :=
  { finite_disjoint_inter_image := by
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        ⊢ ∀ {K L : Set G}, IsCompact K → IsCompact L → (setOf fun γ => Ne (Inter.inter …
      -/
      intro K L hK hL
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        ⊢ (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K) L)  …
      -/
      have : Continuous fun p : G × G => (p.1⁻¹, p.2) := continuous_inv.prodMap continuous_id
      have H : Set.Finite _ :=
        hS ((hK.prod hL).image (continuous_mul.comp this)).compl_mem_cocompact
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        this : Continuous fun p => { fst := Inv.inv p.1, snd := p.2 }
        H : (HasCompl.compl (Set.preimage (⇑S.subtype) (HasCompl.compl (Set.image (Fun …
        ⊢ (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K) L)  …
      -/
      simp only [preimage_compl, compl_compl, coeSubtype, comp_apply] at H
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        this : Continuous fun p => { fst := Inv.inv p.1, snd := p.2 }
        H : (Set.preimage Subtype.val (Set.image (fun a => HMul.hMul (Inv.inv a.1) a.2 …
        ⊢ (setOf fun γ => Ne (Inter.inter (Set.image (fun x => HSMul.hSMul γ x) K) L)  …
      -/
      apply Finite.of_preimage _ (equivOp S).surjective
      /-
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        this : Continuous fun p => { fst := Inv.inv p.1, snd := p.2 }
        H : (Set.preimage Subtype.val (Set.image (fun a => HMul.hMul (Inv.inv a.1) a.2 …
        ⊢ (Set.preimage (⇑S.equivOp) (setOf fun γ => Ne (Inter.inter (Set.image (fun x …
      -/
      convert H using 1
      /-
        case h.e'_2
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        this : Continuous fun p => { fst := Inv.inv p.1, snd := p.2 }
        H : (Set.preimage Subtype.val (Set.image (fun a => HMul.hMul (Inv.inv a.1) a.2 …
        ⊢ Eq (Set.preimage (⇑S.equivOp) (setOf fun γ => Ne (Inter.inter (Set.image (fu …
      -/
      ext x
      /-
        case h.e'_2.h
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        this : Continuous fun p => { fst := Inv.inv p.1, snd := p.2 }
        H : (Set.preimage Subtype.val (Set.image (fun a => HMul.hMul (Inv.inv a.1) a.2 …
        x : Subtype fun x => Membership.mem S x
        ⊢ Iff (Membership.mem (Set.preimage (⇑S.equivOp) (setOf fun γ => Ne (Inter.int …
      -/
      simp only [image_smul, mem_setOf_eq, coeSubtype, mem_preimage, mem_image, Prod.exists]
      /-
        case h.e'_2.h
        G : Type w
        inst✝² : TopologicalSpace G
        inst✝¹ : Group G
        inst✝ : TopologicalGroup G
        S : Subgroup G
        hS : Filter.Tendsto (⇑S.subtype) Filter.cofinite (Filter.cocompact G)
        K L : Set G
        hK : IsCompact K
        hL : IsCompact L
        this : Continuous fun p => { fst := Inv.inv p.1, snd := p.2 }
        H : (Set.preimage Subtype.val (Set.image (fun a => HMul.hMul (Inv.inv a.1) a.2 …
        x : Subtype fun x => Membership.mem S x
        ⊢ Iff (Ne (Inter.inter (HSMul.hSMul (S.equivOp x) K) L) EmptyCollection.emptyC …
      -/
      exact Set.op_smul_inter_ne_empty_iff }
      /-
        🎉 no goals
      -/


/-- Given a compact set `K` inside an open set `U`, there is an open neighborhood `V` of `1`
  such that `K * V ⊆ U`. -/
@[to_additive
  "Given a compact set `K` inside an open set `U`, there is an open neighborhood `V` of
  `0` such that `K + V ⊆ U`."]
theorem compact_open_separated_mul_right {K U : Set G} (hK : IsCompact K) (hU : IsOpen U)
    (hKU : K ⊆ U) : ∃ V ∈ 𝓝 (1 : G), K * V ⊆ U := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : MulOneClass G
    inst✝ : ContinuousMul G
    K U : Set G
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (HasSubset.Subset (HMul.hMul …
  -/
  refine hK.induction_on ?_ ?_ ?_ ?_
    /-
      case refine_1
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (HasSubset.Subset (HMul.hMul …
    -/
  · exact ⟨univ, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      ⊢ ∀ ⦃s t : Set G⦄, HasSubset.Subset s t → (Exists fun V => And (Membership.mem …
    -/
  · rintro s t hst ⟨V, hV, hV'⟩
    /-
      case refine_2.intro.intro
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      s t : Set G
      hst : HasSubset.Subset s t
      V : Set G
      hV : Membership.mem (nhds 1) V
      hV' : HasSubset.Subset (HMul.hMul t V) U
      ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (HasSubset.Subset (HMul.hMul …
    -/
    exact ⟨V, hV, (mul_subset_mul_right hst).trans hV'⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      ⊢ ∀ ⦃s t : Set G⦄, (Exists fun V => And (Membership.mem (nhds 1) V) (HasSubset …
    -/
  · rintro s t ⟨V, V_in, hV'⟩ ⟨W, W_in, hW'⟩
    /-
      case refine_3.intro.intro.intro.intro
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      s t V : Set G
      V_in : Membership.mem (nhds 1) V
      hV' : HasSubset.Subset (HMul.hMul s V) U
      W : Set G
      W_in : Membership.mem (nhds 1) W
      hW' : HasSubset.Subset (HMul.hMul t W) U
      ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (HasSubset.Subset (HMul.hMul …
    -/
    use V ∩ W, inter_mem V_in W_in
    /-
      case right
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      s t V : Set G
      V_in : Membership.mem (nhds 1) V
      hV' : HasSubset.Subset (HMul.hMul s V) U
      W : Set G
      W_in : Membership.mem (nhds 1) W
      hW' : HasSubset.Subset (HMul.hMul t W) U
      ⊢ HasSubset.Subset (HMul.hMul (Union.union s t) (Inter.inter V W)) U
    -/
    rw [union_mul]
    exact
      union_subset ((mul_subset_mul_left V.inter_subset_left).trans hV')
        ((mul_subset_mul_left V.inter_subset_right).trans hW')
    /-
      case refine_4
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      ⊢ ∀ (x : G), Membership.mem K x → Exists fun t => And (Membership.mem (nhdsWit …
    -/
  · intro x hx
    /-
      case refine_4
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      x : G
      hx : Membership.mem K x
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Exists fun V => And …
    -/
    have := tendsto_mul (show U ∈ 𝓝 (x * 1) by simpa using hU.mem_nhds (hKU hx))
    /-
      case refine_4
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      x : G
      hx : Membership.mem K x
      this : Membership.mem (Filter.map (fun p => HMul.hMul p.1 p.2) (nhds { fst :=  …
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Exists fun V => And …
    -/
    rw [nhds_prod_eq, mem_map, mem_prod_iff] at this
    /-
      case refine_4
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      x : G
      hx : Membership.mem K x
      this : Exists fun t₁ => And (Membership.mem (nhds x) t₁) (Exists fun t₂ => And …
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Exists fun V => And …
    -/
    rcases this with ⟨t, ht, s, hs, h⟩
    /-
      case refine_4.intro.intro.intro.intro
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      x : G
      hx : Membership.mem K x
      t : Set G
      ht : Membership.mem (nhds x) t
      s : Set G
      hs : Membership.mem (nhds 1) s
      h : HasSubset.Subset (SProd.sprod t s) (Set.preimage (fun p => HMul.hMul p.1 p …
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Exists fun V => And …
    -/
    rw [← image_subset_iff, image_mul_prod] at h
    /-
      case refine_4.intro.intro.intro.intro
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : MulOneClass G
      inst✝ : ContinuousMul G
      K U : Set G
      hK : IsCompact K
      hU : IsOpen U
      hKU : HasSubset.Subset K U
      x : G
      hx : Membership.mem K x
      t : Set G
      ht : Membership.mem (nhds x) t
      s : Set G
      hs : Membership.mem (nhds 1) s
      h : HasSubset.Subset (HMul.hMul t s) U
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Exists fun V => And …
    -/
    exact ⟨t, mem_nhdsWithin_of_mem_nhds ht, s, hs, h⟩
    /-
      🎉 no goals
    -/


/-- Given a compact set `K` inside an open set `U`, there is an open neighborhood `V` of `1`
  such that `V * K ⊆ U`. -/
@[to_additive
  "Given a compact set `K` inside an open set `U`, there is an open neighborhood `V` of
  `0` such that `V + K ⊆ U`."]
theorem compact_open_separated_mul_left {K U : Set G} (hK : IsCompact K) (hU : IsOpen U)
    (hKU : K ⊆ U) : ∃ V ∈ 𝓝 (1 : G), V * K ⊆ U := by
  rcases compact_open_separated_mul_right (hK.image continuous_op) (opHomeomorph.isOpenMap U hU)
      (image_subset op hKU) with
    ⟨V, hV : V ∈ 𝓝 (op (1 : G)), hV' : op '' K * V ⊆ op '' U⟩
  /-
    case intro.intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : MulOneClass G
    inst✝ : ContinuousMul G
    K U : Set G
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    V : Set (MulOpposite G)
    hV : Membership.mem (nhds (MulOpposite.op 1)) V
    hV' : HasSubset.Subset (HMul.hMul (Set.image MulOpposite.op K) V) (Set.image M …
    ⊢ Exists fun V => And (Membership.mem (nhds 1) V) (HasSubset.Subset (HMul.hMul …
  -/
  refine ⟨op ⁻¹' V, continuous_op.continuousAt hV, ?_⟩
  rwa [← image_preimage_eq V op_surjective, ← image_op_mul, image_subset_iff,
    preimage_image_eq _ op_injective] at hV'


/-- A compact set is covered by finitely many left multiplicative translates of a set
  with non-empty interior. -/
@[to_additive
  "A compact set is covered by finitely many left additive translates of a set
    with non-empty interior."]
theorem compact_covered_by_mul_left_translates {K V : Set G} (hK : IsCompact K)
    (hV : (interior V).Nonempty) : ∃ t : Finset G, K ⊆ ⋃ g ∈ t, (g * ·) ⁻¹' V := by
  obtain ⟨t, ht⟩ : ∃ t : Finset G, K ⊆ ⋃ x ∈ t, interior ((x * ·) ⁻¹' V) := by
    refine
      hK.elim_finite_subcover (fun x => interior <| (x * ·) ⁻¹' V) (fun x => isOpen_interior) ?_
    cases' hV with g₀ hg₀
    refine fun g _ => mem_iUnion.2 ⟨g₀ * g⁻¹, ?_⟩
    refine preimage_interior_subset_interior_preimage (continuous_const.mul continuous_id) ?_
    rwa [mem_preimage, Function.id_def, inv_mul_cancel_right]
  /-
    case intro
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    K V : Set G
    hK : IsCompact K
    hV : (interior V).Nonempty
    t : Finset G
    ht : HasSubset.Subset K (Set.iUnion fun x => Set.iUnion fun h => interior (Set …
    ⊢ Exists fun t => HasSubset.Subset K (Set.iUnion fun g => Set.iUnion fun h =>  …
  -/
  exact ⟨t, Subset.trans ht <| iUnion₂_mono fun g _ => interior_subset⟩
  /-
    🎉 no goals
  -/


/-- Every weakly locally compact separable topological group is σ-compact.
  Note: this is not true if we drop the topological group hypothesis. -/
@[to_additive SeparableWeaklyLocallyCompactAddGroup.sigmaCompactSpace
  "Every weakly locally compact separable topological additive group is σ-compact.
  Note: this is not true if we drop the topological group hypothesis."]
instance (priority := 100) SeparableWeaklyLocallyCompactGroup.sigmaCompactSpace [SeparableSpace G]
    [WeaklyLocallyCompactSpace G] : SigmaCompactSpace G := by
  /-
    G : Type w
    H : Type x
    α : Type u
    β : Type v
    inst✝⁴ : TopologicalSpace G
    inst✝³ : Group G
    inst✝² : TopologicalGroup G
    inst✝¹ : TopologicalSpace.SeparableSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    ⊢ SigmaCompactSpace G
  -/
  obtain ⟨L, hLc, hL1⟩ := exists_compact_mem_nhds (1 : G)
  /-
    case intro.intro
    G : Type w
    H : Type x
    α : Type u
    β : Type v
    inst✝⁴ : TopologicalSpace G
    inst✝³ : Group G
    inst✝² : TopologicalGroup G
    inst✝¹ : TopologicalSpace.SeparableSpace G
    inst✝ : WeaklyLocallyCompactSpace G
    L : Set G
    hLc : IsCompact L
    hL1 : Membership.mem (nhds 1) L
    ⊢ SigmaCompactSpace G
  -/
  refine ⟨⟨fun n => (fun x => x * denseSeq G n) ⁻¹' L, ?_, ?_⟩⟩
    /-
      case intro.intro.refine_1
      G : Type w
      H : Type x
      α : Type u
      β : Type v
      inst✝⁴ : TopologicalSpace G
      inst✝³ : Group G
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalSpace.SeparableSpace G
      inst✝ : WeaklyLocallyCompactSpace G
      L : Set G
      hLc : IsCompact L
      hL1 : Membership.mem (nhds 1) L
      ⊢ ∀ (n : Nat), IsCompact ((fun n => Set.preimage (fun x => HMul.hMul x (Topolo …
    -/
  · intro n
    /-
      case intro.intro.refine_1
      G : Type w
      H : Type x
      α : Type u
      β : Type v
      inst✝⁴ : TopologicalSpace G
      inst✝³ : Group G
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalSpace.SeparableSpace G
      inst✝ : WeaklyLocallyCompactSpace G
      L : Set G
      hLc : IsCompact L
      hL1 : Membership.mem (nhds 1) L
      n : Nat
      ⊢ IsCompact ((fun n => Set.preimage (fun x => HMul.hMul x (TopologicalSpace.de …
    -/
    exact (Homeomorph.mulRight _).isCompact_preimage.mpr hLc
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      G : Type w
      H : Type x
      α : Type u
      β : Type v
      inst✝⁴ : TopologicalSpace G
      inst✝³ : Group G
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalSpace.SeparableSpace G
      inst✝ : WeaklyLocallyCompactSpace G
      L : Set G
      hLc : IsCompact L
      hL1 : Membership.mem (nhds 1) L
      ⊢ Eq (Set.iUnion fun n => (fun n => Set.preimage (fun x => HMul.hMul x (Topolo …
    -/
  · refine iUnion_eq_univ_iff.2 fun x => ?_
    obtain ⟨_, ⟨n, rfl⟩, hn⟩ : (range (denseSeq G) ∩ (fun y => x * y) ⁻¹' L).Nonempty := by
      rw [← (Homeomorph.mulLeft x).apply_symm_apply 1] at hL1
      exact (denseRange_denseSeq G).inter_nhds_nonempty
          ((Homeomorph.mulLeft x).continuous.continuousAt <| hL1)
    /-
      case intro.intro.refine_2.intro.intro.intro
      G : Type w
      H : Type x
      α : Type u
      β : Type v
      inst✝⁴ : TopologicalSpace G
      inst✝³ : Group G
      inst✝² : TopologicalGroup G
      inst✝¹ : TopologicalSpace.SeparableSpace G
      inst✝ : WeaklyLocallyCompactSpace G
      L : Set G
      hLc : IsCompact L
      hL1 : Membership.mem (nhds 1) L
      x : G
      n : Nat
      hn : Membership.mem (Set.preimage (fun y => HMul.hMul x y) L) (TopologicalSpac …
      ⊢ Exists fun i => Membership.mem (Set.preimage (fun x => HMul.hMul x (Topologi …
    -/
    exact ⟨n, hn⟩
    /-
      🎉 no goals
    -/


/-- Given two compact sets in a noncompact topological group, there is a translate of the second
one that is disjoint from the first one. -/
@[to_additive
  "Given two compact sets in a noncompact additive topological group, there is a
  translate of the second one that is disjoint from the first one."]
theorem exists_disjoint_smul_of_isCompact [NoncompactSpace G] {K L : Set G} (hK : IsCompact K)
    (hL : IsCompact L) : ∃ g : G, Disjoint K (g • L) := by
  /-
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : NoncompactSpace G
    K L : Set G
    hK : IsCompact K
    hL : IsCompact L
    ⊢ Exists fun g => Disjoint K (HSMul.hSMul g L)
  -/
  have A : ¬K * L⁻¹ = univ := (hK.mul hL.inv).ne_univ
  obtain ⟨g, hg⟩ : ∃ g, g ∉ K * L⁻¹ := by
    contrapose! A
    exact eq_univ_iff_forall.2 A
  /-
    case intro
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : NoncompactSpace G
    K L : Set G
    hK : IsCompact K
    hL : IsCompact L
    A : Not (Eq (HMul.hMul K (Inv.inv L)) Set.univ)
    g : G
    hg : Not (Membership.mem (HMul.hMul K (Inv.inv L)) g)
    ⊢ Exists fun g => Disjoint K (HSMul.hSMul g L)
  -/
  refine ⟨g, ?_⟩
  /-
    case intro
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : NoncompactSpace G
    K L : Set G
    hK : IsCompact K
    hL : IsCompact L
    A : Not (Eq (HMul.hMul K (Inv.inv L)) Set.univ)
    g : G
    hg : Not (Membership.mem (HMul.hMul K (Inv.inv L)) g)
    ⊢ Disjoint K (HSMul.hSMul g L)
  -/
  refine disjoint_left.2 fun a ha h'a => hg ?_
  /-
    case intro
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : NoncompactSpace G
    K L : Set G
    hK : IsCompact K
    hL : IsCompact L
    A : Not (Eq (HMul.hMul K (Inv.inv L)) Set.univ)
    g : G
    hg : Not (Membership.mem (HMul.hMul K (Inv.inv L)) g)
    a : G
    ha : Membership.mem K a
    h'a : Membership.mem (HSMul.hSMul g L) a
    ⊢ Membership.mem (HMul.hMul K (Inv.inv L)) g
  -/
  rcases h'a with ⟨b, bL, rfl⟩
  /-
    case intro.intro.intro
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : NoncompactSpace G
    K L : Set G
    hK : IsCompact K
    hL : IsCompact L
    A : Not (Eq (HMul.hMul K (Inv.inv L)) Set.univ)
    g : G
    hg : Not (Membership.mem (HMul.hMul K (Inv.inv L)) g)
    b : G
    bL : Membership.mem L b
    ha : Membership.mem K ((fun x => HSMul.hSMul g x) b)
    ⊢ Membership.mem (HMul.hMul K (Inv.inv L)) g
  -/
  refine ⟨g * b, ha, b⁻¹, by simpa only [Set.mem_inv, inv_inv] using bL, ?_⟩
  /-
    case intro.intro.intro
    G : Type w
    inst✝³ : TopologicalSpace G
    inst✝² : Group G
    inst✝¹ : TopologicalGroup G
    inst✝ : NoncompactSpace G
    K L : Set G
    hK : IsCompact K
    hL : IsCompact L
    A : Not (Eq (HMul.hMul K (Inv.inv L)) Set.univ)
    g : G
    hg : Not (Membership.mem (HMul.hMul K (Inv.inv L)) g)
    b : G
    bL : Membership.mem L b
    ha : Membership.mem K ((fun x => HSMul.hSMul g x) b)
    ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (HMul.hMul g b) (Inv.inv b)) g
  -/
  simp only [smul_eq_mul, mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


/-- If a point in a topological group has a compact neighborhood, then the group is
locally compact. -/
@[to_additive]
theorem IsCompact.locallyCompactSpace_of_mem_nhds_of_group {K : Set G} (hK : IsCompact K) {x : G}
    (h : K ∈ 𝓝 x) : LocallyCompactSpace G := by
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    x : G
    h : Membership.mem (nhds x) K
    ⊢ LocallyCompactSpace G
  -/
  suffices WeaklyLocallyCompactSpace G from inferInstance
  /-
    G : Type w
    inst✝² : TopologicalSpace G
    inst✝¹ : Group G
    inst✝ : TopologicalGroup G
    K : Set G
    hK : IsCompact K
    x : G
    h : Membership.mem (nhds x) K
    ⊢ WeaklyLocallyCompactSpace G
  -/
  refine ⟨fun y ↦ ⟨(y * x⁻¹) • K, ?_, ?_⟩⟩
    /-
      case refine_1
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      K : Set G
      hK : IsCompact K
      x : G
      h : Membership.mem (nhds x) K
      y : G
      ⊢ IsCompact (HSMul.hSMul (HMul.hMul y (Inv.inv x)) K)
    -/
  · exact hK.smul _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      K : Set G
      hK : IsCompact K
      x : G
      h : Membership.mem (nhds x) K
      y : G
      ⊢ Membership.mem (nhds y) (HSMul.hSMul (HMul.hMul y (Inv.inv x)) K)
    -/
  · rw [← preimage_smul_inv]
    /-
      case refine_2
      G : Type w
      inst✝² : TopologicalSpace G
      inst✝¹ : Group G
      inst✝ : TopologicalGroup G
      K : Set G
      hK : IsCompact K
      x : G
      h : Membership.mem (nhds x) K
      y : G
      ⊢ Membership.mem (nhds y) (Set.preimage (fun x_1 => HSMul.hSMul (Inv.inv (HMul …
    -/
    exact (continuous_const_smul _).continuousAt.preimage_mem_nhds (by simpa using h)
    /-
      🎉 no goals
    -/


/-- If a function defined on a topological group has a support contained in a
compact set, then either the function is trivial or the group is locally compact. -/
@[to_additive
      "If a function defined on a topological additive group has a support contained in a compact
      set, then either the function is trivial or the group is locally compact."]
theorem eq_zero_or_locallyCompactSpace_of_support_subset_isCompact_of_group
    [TopologicalSpace α] [Zero α] [T1Space α]
    {f : G → α} {k : Set G} (hk : IsCompact k) (hf : support f ⊆ k) (h'f : Continuous f) :
    f = 0 ∨ LocallyCompactSpace G := by
  /-
    G : Type w
    α : Type u
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero α
    inst✝ : T1Space α
    f : G → α
    k : Set G
    hk : IsCompact k
    hf : HasSubset.Subset (Function.support f) k
    h'f : Continuous f
    ⊢ Or (Eq f 0) (LocallyCompactSpace G)
  -/
  refine or_iff_not_imp_left.mpr fun h => ?_
  /-
    G : Type w
    α : Type u
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero α
    inst✝ : T1Space α
    f : G → α
    k : Set G
    hk : IsCompact k
    hf : HasSubset.Subset (Function.support f) k
    h'f : Continuous f
    h : Not (Eq f 0)
    ⊢ LocallyCompactSpace G
  -/
  simp_rw [funext_iff, Pi.zero_apply] at h
  /-
    G : Type w
    α : Type u
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero α
    inst✝ : T1Space α
    f : G → α
    k : Set G
    hk : IsCompact k
    hf : HasSubset.Subset (Function.support f) k
    h'f : Continuous f
    h : Not (∀ (x : G), Eq (f x) 0)
    ⊢ LocallyCompactSpace G
  -/
  push_neg at h
  /-
    G : Type w
    α : Type u
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero α
    inst✝ : T1Space α
    f : G → α
    k : Set G
    hk : IsCompact k
    hf : HasSubset.Subset (Function.support f) k
    h'f : Continuous f
    h : Exists fun x => Ne (f x) 0
    ⊢ LocallyCompactSpace G
  -/
  obtain ⟨x, hx⟩ : ∃ x, f x ≠ 0 := h
  have : k ∈ 𝓝 x :=
    mem_of_superset (h'f.isOpen_support.mem_nhds hx) hf
  /-
    case intro
    G : Type w
    α : Type u
    inst✝⁵ : TopologicalSpace G
    inst✝⁴ : Group G
    inst✝³ : TopologicalGroup G
    inst✝² : TopologicalSpace α
    inst✝¹ : Zero α
    inst✝ : T1Space α
    f : G → α
    k : Set G
    hk : IsCompact k
    hf : HasSubset.Subset (Function.support f) k
    h'f : Continuous f
    x : G
    hx : Ne (f x) 0
    this : Membership.mem (nhds x) k
    ⊢ LocallyCompactSpace G
  -/
  exact IsCompact.locallyCompactSpace_of_mem_nhds_of_group hk this
  /-
    🎉 no goals
  -/


/-- If a function defined on a topological group has compact support, then either
the function is trivial or the group is locally compact. -/
@[to_additive
      "If a function defined on a topological additive group has compact support,
      then either the function is trivial or the group is locally compact."]
theorem HasCompactSupport.eq_zero_or_locallyCompactSpace_of_group
    [TopologicalSpace α] [Zero α] [T1Space α]
    {f : G → α} (hf : HasCompactSupport f) (h'f : Continuous f) :
    f = 0 ∨ LocallyCompactSpace G :=
  eq_zero_or_locallyCompactSpace_of_support_subset_isCompact_of_group hf (subset_tsupport f) h'f


@[to_additive]
theorem nhds_mul (x y : G) : 𝓝 (x * y) = 𝓝 x * 𝓝 y :=
  calc
                                                            /-
                                                              G : Type w
                                                              inst✝² : TopologicalSpace G
                                                              inst✝¹ : Group G
                                                              inst✝ : TopologicalGroup G
                                                              x y : G
                                                              ⊢ Eq (nhds (HMul.hMul x y)) (Filter.map (fun x_1 => HMul.hMul x x_1) (Filter.m …
                                                            -/
    𝓝 (x * y) = map (x * ·) (map (· * y) (𝓝 1 * 𝓝 1)) := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              G : Type w
                                                              inst✝² : TopologicalSpace G
                                                              inst✝¹ : Group G
                                                              inst✝ : TopologicalGroup G
                                                              x y : G
                                                              ⊢ Eq (Filter.map (fun x_1 => HMul.hMul x x_1) (Filter.map (fun x => HMul.hMul  …
                                                            -/
    _ = map₂ (fun a b => x * (a * b * y)) (𝓝 1) (𝓝 1) := by rw [← map₂_mul, map_map₂, map_map₂]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              G : Type w
                                                              inst✝² : TopologicalSpace G
                                                              inst✝¹ : Group G
                                                              inst✝ : TopologicalGroup G
                                                              x y : G
                                                              ⊢ Eq (Filter.map₂ (fun a b => HMul.hMul x (HMul.hMul (HMul.hMul a b) y)) (nhds …
                                                            -/
    _ = map₂ (fun a b => x * a * (b * y)) (𝓝 1) (𝓝 1) := by simp only [mul_assoc]
                                                            /-
                                                              🎉 no goals
                                                            -/
    _ = 𝓝 x * 𝓝 y := by
      rw [← map_mul_left_nhds_one x, ← map_mul_right_nhds_one y, ← map₂_mul, map₂_map_left,
        map₂_map_right]


/-- On a topological group, `𝓝 : G → Filter G` can be promoted to a `MulHom`. -/
@[to_additive (attr := simps)
  "On an additive topological group, `𝓝 : G → Filter G` can be promoted to an `AddHom`."]
def nhdsMulHom : G →ₙ* Filter G where
  toFun := 𝓝
  map_mul' _ _ := nhds_mul _ _


instance {G} [TopologicalSpace G] [Group G] [TopologicalGroup G] :
    TopologicalAddGroup (Additive G) where
  continuous_neg := @continuous_inv G _ _ _


instance {G} [TopologicalSpace G] [AddGroup G] [TopologicalAddGroup G] :
    TopologicalGroup (Multiplicative G) where
  continuous_inv := @continuous_neg G _ _ _


/-- If `G` is a group with topological `⁻¹`, then it is homeomorphic to its units. -/
@[to_additive " If `G` is an additive group with topological negation, then it is homeomorphic to
its additive units."]
def toUnits_homeomorph [Group G] [TopologicalSpace G] [ContinuousInv G] : G ≃ₜ Gˣ where
  toEquiv := toUnits.toEquiv
  continuous_toFun := Units.continuous_iff.2 ⟨continuous_id, continuous_inv⟩
  continuous_invFun := Units.continuous_val


@[to_additive] theorem Units.isEmbedding_val [Group G] [TopologicalSpace G] [ContinuousInv G] :
    IsEmbedding (val : Gˣ → G) :=
  toUnits_homeomorph.symm.isEmbedding


@[deprecated (since := "2024-10-26")]
alias Units.embedding_val := Units.isEmbedding_val


@[to_additive]
instance [ContinuousMul α] : TopologicalGroup αˣ where
  continuous_inv := Units.continuous_iff.2 <| ⟨continuous_coe_inv, continuous_val⟩


/-- The topological group isomorphism between the units of a product of two monoids, and the product
of the units of each monoid. -/
@[to_additive
  "The topological group isomorphism between the additive units of a product of two
  additive monoids, and the product of the additive units of each additive monoid."]
def Homeomorph.prodUnits : (α × β)ˣ ≃ₜ αˣ × βˣ where
  continuous_toFun :=
    (continuous_fst.units_map (MonoidHom.fst α β)).prod_mk
      (continuous_snd.units_map (MonoidHom.snd α β))
  continuous_invFun :=
    Units.continuous_iff.2
      ⟨continuous_val.fst'.prod_mk continuous_val.snd',
        continuous_coe_inv.fst'.prod_mk continuous_coe_inv.snd'⟩
  toEquiv := MulEquiv.prodUnits.toEquiv


@[to_additive]
theorem topologicalGroup_sInf {ts : Set (TopologicalSpace G)}
    (h : ∀ t ∈ ts, @TopologicalGroup G t _) : @TopologicalGroup G (sInf ts) _ :=
  letI := sInf ts
  { toContinuousInv :=
      @continuousInv_sInf _ _ _ fun t ht => @TopologicalGroup.toContinuousInv G t _ <| h t ht
    toContinuousMul :=
      @continuousMul_sInf _ _ _ fun t ht =>
        @TopologicalGroup.toContinuousMul G t _ <| h t ht }


@[to_additive]
theorem topologicalGroup_iInf {ts' : ι → TopologicalSpace G}
    (h' : ∀ i, @TopologicalGroup G (ts' i) _) : @TopologicalGroup G (⨅ i, ts' i) _ := by
  /-
    G : Type w
    ι : Sort u_1
    inst✝ : Group G
    ts' : ι → TopologicalSpace G
    h' : ∀ (i : ι), TopologicalGroup G
    ⊢ TopologicalGroup G
  -/
  rw [← sInf_range]
  /-
    G : Type w
    ι : Sort u_1
    inst✝ : Group G
    ts' : ι → TopologicalSpace G
    h' : ∀ (i : ι), TopologicalGroup G
    ⊢ TopologicalGroup G
  -/
  exact topologicalGroup_sInf (Set.forall_mem_range.mpr h')
  /-
    🎉 no goals
  -/


@[to_additive]
theorem topologicalGroup_inf {t₁ t₂ : TopologicalSpace G} (h₁ : @TopologicalGroup G t₁ _)
    (h₂ : @TopologicalGroup G t₂ _) : @TopologicalGroup G (t₁ ⊓ t₂) _ := by
  /-
    G : Type w
    inst✝ : Group G
    t₁ t₂ : TopologicalSpace G
    h₁ : TopologicalGroup G
    h₂ : TopologicalGroup G
    ⊢ TopologicalGroup G
  -/
  rw [inf_eq_iInf]
  /-
    G : Type w
    inst✝ : Group G
    t₁ t₂ : TopologicalSpace G
    h₁ : TopologicalGroup G
    h₂ : TopologicalGroup G
    ⊢ TopologicalGroup G
  -/
  refine topologicalGroup_iInf fun b => ?_
  /-
    G : Type w
    inst✝ : Group G
    t₁ t₂ : TopologicalSpace G
    h₁ : TopologicalGroup G
    h₂ : TopologicalGroup G
    b : Bool
    ⊢ TopologicalGroup G
  -/
              /-
                🎉 no goals
              -/
  cases b <;> assumption
              /-
                🎉 no goals
              -/


/-- A group topology on a group `α` is a topology for which multiplication and inversion
are continuous. -/
structure GroupTopology (α : Type u) [Group α] extends TopologicalSpace α, TopologicalGroup α :
  Type u


/-- An additive group topology on an additive group `α` is a topology for which addition and
  negation are continuous. -/
structure AddGroupTopology (α : Type u) [AddGroup α] extends TopologicalSpace α,
  TopologicalAddGroup α : Type u


/-- A version of the global `continuous_mul` suitable for dot notation. -/
@[to_additive "A version of the global `continuous_add` suitable for dot notation."]
theorem continuous_mul' (g : GroupTopology α) :
    haveI := g.toTopologicalSpace
    Continuous fun p : α × α => p.1 * p.2 := by
  /-
    α : Type u
    inst✝ : Group α
    g : GroupTopology α
    ⊢ Continuous fun p => HMul.hMul p.1 p.2
  -/
  letI := g.toTopologicalSpace
  /-
    α : Type u
    inst✝ : Group α
    g : GroupTopology α
    this : TopologicalSpace α := g.toTopologicalSpace
    ⊢ Continuous fun p => HMul.hMul p.1 p.2
  -/
  haveI := g.toTopologicalGroup
  /-
    α : Type u
    inst✝ : Group α
    g : GroupTopology α
    this✝ : TopologicalSpace α := g.toTopologicalSpace
    this : TopologicalGroup α
    ⊢ Continuous fun p => HMul.hMul p.1 p.2
  -/
  exact continuous_mul
  /-
    🎉 no goals
  -/


/-- A version of the global `continuous_inv` suitable for dot notation. -/
@[to_additive "A version of the global `continuous_neg` suitable for dot notation."]
theorem continuous_inv' (g : GroupTopology α) :
    haveI := g.toTopologicalSpace
    Continuous (Inv.inv : α → α) := by
  /-
    α : Type u
    inst✝ : Group α
    g : GroupTopology α
    ⊢ Continuous Inv.inv
  -/
  letI := g.toTopologicalSpace
  /-
    α : Type u
    inst✝ : Group α
    g : GroupTopology α
    this : TopologicalSpace α := g.toTopologicalSpace
    ⊢ Continuous Inv.inv
  -/
  haveI := g.toTopologicalGroup
  /-
    α : Type u
    inst✝ : Group α
    g : GroupTopology α
    this✝ : TopologicalSpace α := g.toTopologicalSpace
    this : TopologicalGroup α
    ⊢ Continuous Inv.inv
  -/
  exact continuous_inv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem toTopologicalSpace_injective :
    Function.Injective (toTopologicalSpace : GroupTopology α → TopologicalSpace α) :=
  fun f g h => by
    /-
      α : Type u
      inst✝ : Group α
      f g : GroupTopology α
      h : Eq f.toTopologicalSpace g.toTopologicalSpace
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      α : Type u
      inst✝ : Group α
      g : GroupTopology α
      toTopologicalSpace✝ : TopologicalSpace α
      toTopologicalGroup✝ : TopologicalGroup α
      h : Eq { toTopologicalSpace := toTopologicalSpace✝, toTopologicalGroup := toTo …
      ⊢ Eq { toTopologicalSpace := toTopologicalSpace✝, toTopologicalGroup := toTopo …
    -/
    cases g
    /-
      case mk.mk
      α : Type u
      inst✝ : Group α
      toTopologicalSpace✝¹ : TopologicalSpace α
      toTopologicalGroup✝¹ : TopologicalGroup α
      toTopologicalSpace✝ : TopologicalSpace α
      toTopologicalGroup✝ : TopologicalGroup α
      h : Eq { toTopologicalSpace := toTopologicalSpace✝¹, toTopologicalGroup := toT …
      ⊢ Eq { toTopologicalSpace := toTopologicalSpace✝¹, toTopologicalGroup := toTop …
    -/
    congr
    /-
      🎉 no goals
    -/


@[to_additive (attr := ext)]
theorem ext' {f g : GroupTopology α} (h : f.IsOpen = g.IsOpen) : f = g :=
  toTopologicalSpace_injective <| TopologicalSpace.ext h


/-- The ordering on group topologies on the group `γ`. `t ≤ s` if every set open in `s` is also open
in `t` (`t` is finer than `s`). -/
@[to_additive
  "The ordering on group topologies on the group `γ`. `t ≤ s` if every set open in `s`
  is also open in `t` (`t` is finer than `s`)."]
instance : PartialOrder (GroupTopology α) :=
  PartialOrder.lift toTopologicalSpace toTopologicalSpace_injective


@[to_additive (attr := simp)]
theorem toTopologicalSpace_le {x y : GroupTopology α} :
    x.toTopologicalSpace ≤ y.toTopologicalSpace ↔ x ≤ y :=
  Iff.rfl


@[to_additive]
instance : Top (GroupTopology α) :=
  let _t : TopologicalSpace α := ⊤
  ⟨{  continuous_mul := continuous_top
      continuous_inv := continuous_top }⟩


@[to_additive (attr := simp)]
theorem toTopologicalSpace_top : (⊤ : GroupTopology α).toTopologicalSpace = ⊤ :=
  rfl


@[to_additive]
instance : Bot (GroupTopology α) :=
  let _t : TopologicalSpace α := ⊥
  ⟨{  continuous_mul := by
        /-
          G : Type w
          H : Type x
          α : Type u
          β : Type v
          inst✝ : Group α
          _t : TopologicalSpace α := Bot.bot
          ⊢ Continuous fun p => HMul.hMul p.1 p.2
        -/
        haveI := discreteTopology_bot α
        /-
          G : Type w
          H : Type x
          α : Type u
          β : Type v
          inst✝ : Group α
          _t : TopologicalSpace α := Bot.bot
          this : DiscreteTopology α
          ⊢ Continuous fun p => HMul.hMul p.1 p.2
        -/
        continuity
        /-
          🎉 no goals
        -/
      continuous_inv := continuous_bot }⟩


@[to_additive (attr := simp)]
theorem toTopologicalSpace_bot : (⊥ : GroupTopology α).toTopologicalSpace = ⊥ :=
  rfl


@[to_additive]
instance : BoundedOrder (GroupTopology α) where
  top := ⊤
  le_top x := show x.toTopologicalSpace ≤ ⊤ from le_top
  bot := ⊥
  bot_le x := show ⊥ ≤ x.toTopologicalSpace from bot_le


@[to_additive]
instance : Min (GroupTopology α) where min x y := ⟨x.1 ⊓ y.1, topologicalGroup_inf x.2 y.2⟩


@[to_additive (attr := simp)]
theorem toTopologicalSpace_inf (x y : GroupTopology α) :
    (x ⊓ y).toTopologicalSpace = x.toTopologicalSpace ⊓ y.toTopologicalSpace :=
  rfl


@[to_additive]
instance : SemilatticeInf (GroupTopology α) :=
  toTopologicalSpace_injective.semilatticeInf _ toTopologicalSpace_inf


@[to_additive]
instance : Inhabited (GroupTopology α) :=
  ⟨⊤⟩


/-- Infimum of a collection of group topologies. -/
@[to_additive "Infimum of a collection of additive group topologies"]
instance : InfSet (GroupTopology α) where
  sInf S :=
    ⟨sInf (toTopologicalSpace '' S), topologicalGroup_sInf <| forall_mem_image.2 fun t _ => t.2⟩


@[to_additive (attr := simp)]
theorem toTopologicalSpace_sInf (s : Set (GroupTopology α)) :
    (sInf s).toTopologicalSpace = sInf (toTopologicalSpace '' s) := rfl


@[to_additive (attr := simp)]
theorem toTopologicalSpace_iInf {ι} (s : ι → GroupTopology α) :
    (⨅ i, s i).toTopologicalSpace = ⨅ i, (s i).toTopologicalSpace :=
  congr_arg sInf (range_comp _ _).symm


/-- Group topologies on `γ` form a complete lattice, with `⊥` the discrete topology and `⊤` the
indiscrete topology.

The infimum of a collection of group topologies is the topology generated by all their open sets
(which is a group topology).

The supremum of two group topologies `s` and `t` is the infimum of the family of all group
topologies contained in the intersection of `s` and `t`. -/
@[to_additive
  "Group topologies on `γ` form a complete lattice, with `⊥` the discrete topology and
  `⊤` the indiscrete topology.

  The infimum of a collection of group topologies is the topology generated by all their open sets
  (which is a group topology).

  The supremum of two group topologies `s` and `t` is the infimum of the family of all group
  topologies contained in the intersection of `s` and `t`."]
instance : CompleteSemilatticeInf (GroupTopology α) :=
  { inferInstanceAs (InfSet (GroupTopology α)),
    inferInstanceAs (PartialOrder (GroupTopology α)) with
    sInf_le := fun _ a haS => toTopologicalSpace_le.1 <| sInf_le ⟨a, haS, rfl⟩
    le_sInf := by
      /-
        G : Type w
        H : Type x
        α : Type u
        β : Type v
        inst✝ : Group α
        ⊢ ∀ (s : Set (GroupTopology α)) (a : GroupTopology α), (∀ (b : GroupTopology α …
      -/
      intro S a hab
      /-
        G : Type w
        H : Type x
        α : Type u
        β : Type v
        inst✝ : Group α
        S : Set (GroupTopology α)
        a : GroupTopology α
        hab : ∀ (b : GroupTopology α), Membership.mem S b → LE.le a b
        ⊢ LE.le a (InfSet.sInf S)
      -/
      apply (inferInstanceAs (CompleteLattice (TopologicalSpace α))).le_sInf
      /-
        case a
        G : Type w
        H : Type x
        α : Type u
        β : Type v
        inst✝ : Group α
        S : Set (GroupTopology α)
        a : GroupTopology α
        hab : ∀ (b : GroupTopology α), Membership.mem S b → LE.le a b
        ⊢ ∀ (b : TopologicalSpace α), Membership.mem (Set.image GroupTopology.toTopolo …
      -/
      rintro _ ⟨b, hbS, rfl⟩
      /-
        case a.intro.intro
        G : Type w
        H : Type x
        α : Type u
        β : Type v
        inst✝ : Group α
        S : Set (GroupTopology α)
        a : GroupTopology α
        hab : ∀ (b : GroupTopology α), Membership.mem S b → LE.le a b
        b : GroupTopology α
        hbS : Membership.mem S b
        ⊢ LE.le a.toTopologicalSpace b.toTopologicalSpace
      -/
      exact hab b hbS }
      /-
        🎉 no goals
      -/


@[to_additive]
instance : CompleteLattice (GroupTopology α) :=
  { inferInstanceAs (BoundedOrder (GroupTopology α)),
    inferInstanceAs (SemilatticeInf (GroupTopology α)),
    completeLatticeOfCompleteSemilatticeInf _ with
    inf := (· ⊓ ·)
    top := ⊤
    bot := ⊥ }


/-- Given `f : α → β` and a topology on `α`, the coinduced group topology on `β` is the finest
topology such that `f` is continuous and `β` is a topological group. -/
@[to_additive
  "Given `f : α → β` and a topology on `α`, the coinduced additive group topology on `β`
  is the finest topology such that `f` is continuous and `β` is a topological additive group."]
def coinduced {α β : Type*} [t : TopologicalSpace α] [Group β] (f : α → β) : GroupTopology β :=
  sInf { b : GroupTopology β | TopologicalSpace.coinduced f t ≤ b.toTopologicalSpace }


@[to_additive]
theorem coinduced_continuous {α β : Type*} [t : TopologicalSpace α] [Group β] (f : α → β) :
    Continuous[t, (coinduced f).toTopologicalSpace] f := by
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    inst✝ : Group β
    f : α → β
    ⊢ Continuous f
  -/
  rw [continuous_sInf_rng]
  /-
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    inst✝ : Group β
    f : α → β
    ⊢ ∀ (t_1 : TopologicalSpace β), Membership.mem (Set.image GroupTopology.toTopo …
  -/
  rintro _ ⟨t', ht', rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    t : TopologicalSpace α
    inst✝ : Group β
    f : α → β
    t' : GroupTopology β
    ht' : Membership.mem (setOf fun b => LE.le (TopologicalSpace.coinduced f t) b. …
    ⊢ Continuous f
  -/
  exact continuous_iff_coinduced_le.2 ht'
  /-
    🎉 no goals
  -/


