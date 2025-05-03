/-- Class `ContinuousSMul M X` says that the scalar multiplication `(•) : M → X → X`
is continuous in both arguments. We use the same class for all kinds of multiplicative actions,
including (semi)modules and algebras. -/
class ContinuousSMul (M X : Type*) [SMul M X] [TopologicalSpace M] [TopologicalSpace X] :
    Prop where
  /-- The scalar multiplication `(•)` is continuous. -/
  continuous_smul : Continuous fun p : M × X => p.1 • p.2


/-- Class `ContinuousVAdd M X` says that the additive action `(+ᵥ) : M → X → X`
is continuous in both arguments. We use the same class for all kinds of additive actions,
including (semi)modules and algebras. -/
class ContinuousVAdd (M X : Type*) [VAdd M X] [TopologicalSpace M] [TopologicalSpace X] :
    Prop where
  /-- The additive action `(+ᵥ)` is continuous. -/
  continuous_vadd : Continuous fun p : M × X => p.1 +ᵥ p.2


lemma IsScalarTower.continuousSMul {M : Type*} (N : Type*) {α : Type*} [Monoid N] [SMul M N]
    [MulAction N α] [SMul M α] [IsScalarTower M N α] [TopologicalSpace M] [TopologicalSpace N]
    [TopologicalSpace α] [ContinuousSMul M N] [ContinuousSMul N α] : ContinuousSMul M α :=
  { continuous_smul := by
      /-
        M : Type u_5
        N : Type u_6
        α : Type u_7
        inst✝⁹ : Monoid N
        inst✝⁸ : SMul M N
        inst✝⁷ : MulAction N α
        inst✝⁶ : SMul M α
        inst✝⁵ : IsScalarTower M N α
        inst✝⁴ : TopologicalSpace M
        inst✝³ : TopologicalSpace N
        inst✝² : TopologicalSpace α
        inst✝¹ : ContinuousSMul M N
        inst✝ : ContinuousSMul N α
        ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
      -/
      suffices Continuous (fun p : M × α ↦ (p.1 • (1 : N)) • p.2) by simpa
      /-
        M : Type u_5
        N : Type u_6
        α : Type u_7
        inst✝⁹ : Monoid N
        inst✝⁸ : SMul M N
        inst✝⁷ : MulAction N α
        inst✝⁶ : SMul M α
        inst✝⁵ : IsScalarTower M N α
        inst✝⁴ : TopologicalSpace M
        inst✝³ : TopologicalSpace N
        inst✝² : TopologicalSpace α
        inst✝¹ : ContinuousSMul M N
        inst✝ : ContinuousSMul N α
        ⊢ Continuous fun p => HSMul.hSMul (HSMul.hSMul p.1 1) p.2
      -/
      fun_prop }
      /-
        🎉 no goals
      -/


@[to_additive]
instance : ContinuousSMul (ULift M) X :=
  ⟨(continuous_smul (M := M)).comp₂ (continuous_uLift_down.comp continuous_fst) continuous_snd⟩


@[to_additive]
instance (priority := 100) ContinuousSMul.continuousConstSMul : ContinuousConstSMul M X where
  continuous_const_smul _ := continuous_smul.comp (continuous_const.prod_mk continuous_id)


theorem ContinuousSMul.induced {R : Type*} {α : Type*} {β : Type*} {F : Type*} [FunLike F α β]
    [Semiring R] [AddCommMonoid α] [AddCommMonoid β] [Module R α] [Module R β]
    [TopologicalSpace R] [LinearMapClass F R α β] [tβ : TopologicalSpace β] [ContinuousSMul R β]
    (f : F) : @ContinuousSMul R α _ _ (tβ.induced f) := by
  /-
    R : Type u_5
    α : Type u_6
    β : Type u_7
    F : Type u_8
    inst✝⁸ : FunLike F α β
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid α
    inst✝⁵ : AddCommMonoid β
    inst✝⁴ : Module R α
    inst✝³ : Module R β
    inst✝² : TopologicalSpace R
    inst✝¹ : LinearMapClass F R α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousSMul R β
    f : F
    ⊢ ContinuousSMul R α
  -/
  let tα := tβ.induced f
  /-
    R : Type u_5
    α : Type u_6
    β : Type u_7
    F : Type u_8
    inst✝⁸ : FunLike F α β
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid α
    inst✝⁵ : AddCommMonoid β
    inst✝⁴ : Module R α
    inst✝³ : Module R β
    inst✝² : TopologicalSpace R
    inst✝¹ : LinearMapClass F R α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousSMul R β
    f : F
    tα : TopologicalSpace α := TopologicalSpace.induced (⇑f) tβ
    ⊢ ContinuousSMul R α
  -/
  refine ⟨continuous_induced_rng.2 ?_⟩
  /-
    R : Type u_5
    α : Type u_6
    β : Type u_7
    F : Type u_8
    inst✝⁸ : FunLike F α β
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid α
    inst✝⁵ : AddCommMonoid β
    inst✝⁴ : Module R α
    inst✝³ : Module R β
    inst✝² : TopologicalSpace R
    inst✝¹ : LinearMapClass F R α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousSMul R β
    f : F
    tα : TopologicalSpace α := TopologicalSpace.induced (⇑f) tβ
    ⊢ Continuous (Function.comp ⇑f fun p => HSMul.hSMul p.1 p.2)
  -/
  simp only [Function.comp_def, map_smul]
  /-
    R : Type u_5
    α : Type u_6
    β : Type u_7
    F : Type u_8
    inst✝⁸ : FunLike F α β
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid α
    inst✝⁵ : AddCommMonoid β
    inst✝⁴ : Module R α
    inst✝³ : Module R β
    inst✝² : TopologicalSpace R
    inst✝¹ : LinearMapClass F R α β
    tβ : TopologicalSpace β
    inst✝ : ContinuousSMul R β
    f : F
    tα : TopologicalSpace α := TopologicalSpace.induced (⇑f) tβ
    ⊢ Continuous fun x => HSMul.hSMul x.1 (f x.2)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Filter.Tendsto.smul {f : α → M} {g : α → X} {l : Filter α} {c : M} {a : X}
    (hf : Tendsto f l (𝓝 c)) (hg : Tendsto g l (𝓝 a)) :
    Tendsto (fun x => f x • g x) l (𝓝 <| c • a) :=
  (continuous_smul.tendsto _).comp (hf.prod_mk_nhds hg)


@[to_additive]
theorem Filter.Tendsto.smul_const {f : α → M} {l : Filter α} {c : M} (hf : Tendsto f l (𝓝 c))
    (a : X) : Tendsto (fun x => f x • a) l (𝓝 (c • a)) :=
  hf.smul tendsto_const_nhds


@[to_additive]
theorem ContinuousWithinAt.smul (hf : ContinuousWithinAt f s b) (hg : ContinuousWithinAt g s b) :
    ContinuousWithinAt (fun x => f x • g x) s b :=
  Filter.Tendsto.smul hf hg


@[to_additive (attr := fun_prop)]
theorem ContinuousAt.smul (hf : ContinuousAt f b) (hg : ContinuousAt g b) :
    ContinuousAt (fun x => f x • g x) b :=
  Filter.Tendsto.smul hf hg


@[to_additive (attr := fun_prop)]
theorem ContinuousOn.smul (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun x => f x • g x) s := fun x hx => (hf x hx).smul (hg x hx)


@[to_additive (attr := continuity, fun_prop)]
theorem Continuous.smul (hf : Continuous f) (hg : Continuous g) : Continuous fun x => f x • g x :=
  continuous_smul.comp (hf.prod_mk hg)


/-- If a scalar action is central, then its right action is continuous when its left action is. -/
@[to_additive "If an additive action is central, then its right action is continuous when its left
action is."]
instance ContinuousSMul.op [SMul Mᵐᵒᵖ X] [IsCentralScalar M X] : ContinuousSMul Mᵐᵒᵖ X :=
  ⟨by
    suffices Continuous fun p : M × X => MulOpposite.op p.fst • p.snd from
      this.comp (MulOpposite.continuous_unop.prodMap continuous_id)
    /-
      M : Type u_1
      X : Type u_2
      Y : Type u_3
      α : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : SMul M X
      inst✝² : ContinuousSMul M X
      f : Y → M
      g : Y → X
      b : Y
      s : Set Y
      inst✝¹ : SMul (MulOpposite M) X
      inst✝ : IsCentralScalar M X
      ⊢ Continuous fun p => HSMul.hSMul (MulOpposite.op p.1) p.2
    -/
    simpa only [op_smul_eq_smul] using (continuous_smul : Continuous fun p : M × X => _)⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance MulOpposite.continuousSMul : ContinuousSMul M Xᵐᵒᵖ :=
  ⟨MulOpposite.continuous_op.comp <|
      continuous_smul.comp <| continuous_id.prodMap MulOpposite.continuous_unop⟩


@[to_additive]
protected theorem Specializes.smul {a b : M} {x y : X} (h₁ : a ⤳ b) (h₂ : x ⤳ y) :
    (a • x) ⤳ (b • y) :=
  (h₁.prod h₂).map continuous_smul


@[to_additive]
protected theorem Inseparable.smul {a b : M} {x y : X} (h₁ : Inseparable a b)
    (h₂ : Inseparable x y) : Inseparable (a • x) (b • y) :=
  (h₁.prod h₂).map continuous_smul


@[to_additive]
lemma IsCompact.smul_set {k : Set M} {u : Set X} (hk : IsCompact k) (hu : IsCompact u) :
    IsCompact (k • u) := by
  /-
    M : Type u_1
    X : Type u_2
    inst✝³ : TopologicalSpace M
    inst✝² : TopologicalSpace X
    inst✝¹ : SMul M X
    inst✝ : ContinuousSMul M X
    k : Set M
    u : Set X
    hk : IsCompact k
    hu : IsCompact u
    ⊢ IsCompact (HSMul.hSMul k u)
  -/
  rw [← Set.image_smul_prod]
  /-
    M : Type u_1
    X : Type u_2
    inst✝³ : TopologicalSpace M
    inst✝² : TopologicalSpace X
    inst✝¹ : SMul M X
    inst✝ : ContinuousSMul M X
    k : Set M
    u : Set X
    hk : IsCompact k
    hu : IsCompact u
    ⊢ IsCompact (Set.image (fun x => HSMul.hSMul x.1 x.2) (SProd.sprod k u))
  -/
  exact IsCompact.image (hk.prod hu) continuous_smul
  /-
    🎉 no goals
  -/


@[to_additive]
lemma smul_set_closure_subset (K : Set M) (L : Set X) :
    closure K • closure L ⊆ closure (K • L) :=
  Set.smul_subset_iff.2 fun _x hx _y hy ↦ map_mem_closure₂ continuous_smul hx hy fun _a ha _b hb ↦
    Set.smul_mem_smul ha hb


/-- Suppose that `N` acts on `X` and `M` continuously acts on `Y`.
Suppose that `g : Y → X` is an action homomorphism in the following sense:
there exists a continuous function `f : N → M` such that `g (c • x) = f c • g x`.
Then the action of `N` on `X` is continuous as well.

In many cases, `f = id` so that `g` is an action homomorphism in the sense of `MulActionHom`.
However, this version also works for semilinear maps and `f = Units.val`. -/
@[to_additive
  "Suppose that `N` additively acts on `X` and `M` continuously additively acts on `Y`.
Suppose that `g : Y → X` is an additive action homomorphism in the following sense:
there exists a continuous function `f : N → M` such that `g (c +ᵥ x) = f c +ᵥ g x`.
Then the action of `N` on `X` is continuous as well.

In many cases, `f = id` so that `g` is an action homomorphism in the sense of `AddActionHom`.
However, this version also works for `f = AddUnits.val`."]
lemma Topology.IsInducing.continuousSMul {N : Type*} [SMul N Y] [TopologicalSpace N] {f : N → M}
    (hg : IsInducing g) (hf : Continuous f) (hsmul : ∀ {c x}, g (c • x) = f c • g x) :
    ContinuousSMul N Y where
  continuous_smul := by
    simpa only [hg.continuous_iff, Function.comp_def, hsmul]
      using (hf.comp continuous_fst).smul <| hg.continuous.comp continuous_snd


@[deprecated (since := "2024-10-28")] alias Inducing.continuousSMul := IsInducing.continuousSMul


@[to_additive]
instance SMulMemClass.continuousSMul {S : Type*} [SetLike S X] [SMulMemClass S M X] (s : S) :
    ContinuousSMul M s :=
  IsInducing.subtypeVal.continuousSMul continuous_id rfl


@[to_additive]
instance Units.continuousSMul : ContinuousSMul Mˣ X :=
  IsInducing.id.continuousSMul Units.continuous_val rfl


/-- If an action is continuous, then composing this action with a continuous homomorphism gives
again a continuous action. -/
@[to_additive]
theorem MulAction.continuousSMul_compHom
    {N : Type*} [TopologicalSpace N] [Monoid N] {f : N →* M} (hf : Continuous f) :
    letI : MulAction N X := MulAction.compHom _ f
    ContinuousSMul N X := by
  /-
    M : Type u_1
    X : Type u_2
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : Monoid M
    inst✝³ : MulAction M X
    inst✝² : ContinuousSMul M X
    N : Type u_5
    inst✝¹ : TopologicalSpace N
    inst✝ : Monoid N
    f : MonoidHom N M
    hf : Continuous ⇑f
    ⊢ ContinuousSMul N X
  -/
  let _ : MulAction N X := MulAction.compHom _ f
  /-
    M : Type u_1
    X : Type u_2
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : Monoid M
    inst✝³ : MulAction M X
    inst✝² : ContinuousSMul M X
    N : Type u_5
    inst✝¹ : TopologicalSpace N
    inst✝ : Monoid N
    f : MonoidHom N M
    hf : Continuous ⇑f
    x✝ : MulAction N X := MulAction.compHom X f
    ⊢ ContinuousSMul N X
  -/
  exact ⟨(hf.comp continuous_fst).smul continuous_snd⟩
  /-
    🎉 no goals
  -/


@[to_additive]
instance Submonoid.continuousSMul {S : Submonoid M} : ContinuousSMul S X :=
  IsInducing.id.continuousSMul continuous_subtype_val rfl


@[to_additive]
instance Subgroup.continuousSMul {S : Subgroup M} : ContinuousSMul S X :=
  S.toSubmonoid.continuousSMul


/-- The stabilizer of a continuous group action on a discrete space is an open subgroup. -/
lemma stabilizer_isOpen [DiscreteTopology X] (x : X) : IsOpen (MulAction.stabilizer M x : Set M) :=
                                           /-
                                             M : Type u_1
                                             X : Type u_2
                                             inst✝⁵ : TopologicalSpace M
                                             inst✝⁴ : TopologicalSpace X
                                             inst✝³ : Group M
                                             inst✝² : MulAction M X
                                             inst✝¹ : ContinuousSMul M X
                                             inst✝ : DiscreteTopology X
                                             x : X
                                             ⊢ Continuous fun g => HSMul.hSMul g x
                                           -/
  IsOpen.preimage (f := fun g ↦ g • x) (by fun_prop) (isOpen_discrete {x})
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive]
instance Prod.continuousSMul [SMul M X] [SMul M Y] [ContinuousSMul M X] [ContinuousSMul M Y] :
    ContinuousSMul M (X × Y) :=
  ⟨(continuous_fst.smul (continuous_fst.comp continuous_snd)).prod_mk
      (continuous_fst.smul (continuous_snd.comp continuous_snd))⟩


@[to_additive]
instance {ι : Type*} {γ : ι → Type*} [∀ i, TopologicalSpace (γ i)] [∀ i, SMul M (γ i)]
    [∀ i, ContinuousSMul M (γ i)] : ContinuousSMul M (∀ i, γ i) :=
  ⟨continuous_pi fun i =>
      (continuous_fst.smul continuous_snd).comp <|
        continuous_fst.prod_mk ((continuous_apply i).comp continuous_snd)⟩


@[to_additive]
theorem continuousSMul_sInf {ts : Set (TopologicalSpace X)}
    (h : ∀ t ∈ ts, @ContinuousSMul M X _ _ t) : @ContinuousSMul M X _ _ (sInf ts) :=
  -- Porting note: {} doesn't work because `sInf ts` isn't found by TC search. `(_)` finds it by
  -- unification instead.
  @ContinuousSMul.mk M X _ _ (_) <| by
      -- Porting note: needs `( :)`
      /-
        M : Type u_2
        X : Type u_3
        inst✝¹ : TopologicalSpace M
        inst✝ : SMul M X
        ts : Set (TopologicalSpace X)
        h : ∀ (t : TopologicalSpace X), Membership.mem ts t → ContinuousSMul M X
        ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
      -/
      rw [← (@sInf_singleton _ _ ‹TopologicalSpace M›:)]
      exact
        continuous_sInf_rng.2 fun t ht =>
          continuous_sInf_dom₂ (Eq.refl _) ht
            (@ContinuousSMul.continuous_smul _ _ _ _ t (h t ht))


@[to_additive]
theorem continuousSMul_iInf {ts' : ι → TopologicalSpace X}
    (h : ∀ i, @ContinuousSMul M X _ _ (ts' i)) : @ContinuousSMul M X _ _ (⨅ i, ts' i) :=
  continuousSMul_sInf <| Set.forall_mem_range.mpr h


@[to_additive]
theorem continuousSMul_inf {t₁ t₂ : TopologicalSpace X} [@ContinuousSMul M X _ _ t₁]
    [@ContinuousSMul M X _ _ t₂] : @ContinuousSMul M X _ _ (t₁ ⊓ t₂) := by
  /-
    M : Type u_2
    X : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : SMul M X
    t₁ t₂ : TopologicalSpace X
    inst✝¹ : ContinuousSMul M X
    inst✝ : ContinuousSMul M X
    ⊢ ContinuousSMul M X
  -/
  rw [inf_eq_iInf]
  /-
    M : Type u_2
    X : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : SMul M X
    t₁ t₂ : TopologicalSpace X
    inst✝¹ : ContinuousSMul M X
    inst✝ : ContinuousSMul M X
    ⊢ ContinuousSMul M X
  -/
  refine continuousSMul_iInf fun b => ?_
  /-
    M : Type u_2
    X : Type u_3
    inst✝³ : TopologicalSpace M
    inst✝² : SMul M X
    t₁ t₂ : TopologicalSpace X
    inst✝¹ : ContinuousSMul M X
    inst✝ : ContinuousSMul M X
    b : Bool
    ⊢ ContinuousSMul M X
  -/
              /-
                🎉 no goals
              -/
  cases b <;> assumption
              /-
                🎉 no goals
              -/


include G in
/-- An `AddTorsor` for a connected space is a connected space. This is not an instance because
it loops for a group as a torsor over itself. -/
protected theorem AddTorsor.connectedSpace : ConnectedSpace P :=
  { isPreconnected_univ := by
      convert
        isPreconnected_univ.image (Equiv.vaddConst (Classical.arbitrary P) : G → P)
          (continuous_id.vadd continuous_const).continuousOn
      /-
        case h.e'_3
        G : Type u_1
        P : Type u_2
        inst✝⁵ : AddGroup G
        inst✝⁴ : AddTorsor G P
        inst✝³ : TopologicalSpace G
        inst✝² : PreconnectedSpace G
        inst✝¹ : TopologicalSpace P
        inst✝ : ContinuousVAdd G P
        ⊢ Eq Set.univ (Set.image (⇑(Equiv.vaddConst (Classical.arbitrary P))) Set.univ)
      -/
      rw [Set.image_univ, Equiv.range_eq_univ]
      /-
        🎉 no goals
      -/
    toNonempty := inferInstance }


