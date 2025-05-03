/-- `C₀(α, β)` is the type of continuous functions `α → β` which vanish at infinity from a
topological space to a metric space with a zero element.

When possible, instead of parametrizing results over `(f : C₀(α, β))`,
you should parametrize over `(F : Type*) [ZeroAtInftyContinuousMapClass F α β] (f : F)`.

When you extend this structure, make sure to extend `ZeroAtInftyContinuousMapClass`. -/
structure ZeroAtInftyContinuousMap (α : Type u) (β : Type v) [TopologicalSpace α] [Zero β]
    [TopologicalSpace β] extends ContinuousMap α β : Type max u v where
  /-- The function tends to zero along the `cocompact` filter. -/
  zero_at_infty' : Tendsto toFun (cocompact α) (𝓝 0)


@[inherit_doc]
scoped[ZeroAtInfty] notation (priority := 2000) "C₀(" α ", " β ")" => ZeroAtInftyContinuousMap α β


@[inherit_doc]
scoped[ZeroAtInfty] notation α " →C₀ " β => ZeroAtInftyContinuousMap α β


/-- `ZeroAtInftyContinuousMapClass F α β` states that `F` is a type of continuous maps which
vanish at infinity.

You should also extend this typeclass when you extend `ZeroAtInftyContinuousMap`. -/
class ZeroAtInftyContinuousMapClass (F : Type*) (α β : outParam Type*) [TopologicalSpace α]
    [Zero β] [TopologicalSpace β] [FunLike F α β] extends ContinuousMapClass F α β : Prop where
  /-- Each member of the class tends to zero along the `cocompact` filter. -/
  zero_at_infty (f : F) : Tendsto f (cocompact α) (𝓝 0)


instance instFunLike : FunLike C₀(α, β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      f g : ZeroAtInftyContinuousMap α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      g : ZeroAtInftyContinuousMap α β
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      zero_at_infty'✝ : Filter.Tendsto { toFun := toFun✝, continuous_toFun := contin …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, continuous_toFun := continuous_t …
      ⊢ Eq { toFun := toFun✝, continuous_toFun := continuous_toFun✝, zero_at_infty'  …
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      toFun✝¹ : α → β
      continuous_toFun✝¹ : Continuous toFun✝¹
      zero_at_infty'✝¹ : Filter.Tendsto { toFun := toFun✝¹, continuous_toFun := cont …
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      zero_at_infty'✝ : Filter.Tendsto { toFun := toFun✝, continuous_toFun := contin …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, continuous_toFun := continuous_ …
      ⊢ Eq { toFun := toFun✝¹, continuous_toFun := continuous_toFun✝¹, zero_at_infty …
    -/
    congr
    /-
      🎉 no goals
    -/


instance instZeroAtInftyContinuousMapClass : ZeroAtInftyContinuousMapClass C₀(α, β) α β where
  map_continuous f := f.continuous_toFun
  zero_at_infty f := f.zero_at_infty'


instance instCoeTC : CoeTC F C₀(α, β) :=
  ⟨fun f =>
    { toFun := f
      continuous_toFun := map_continuous f
      zero_at_infty' := zero_at_infty f }⟩


@[simp]
theorem coe_toContinuousMap (f : C₀(α, β)) : (f.toContinuousMap : α → β) = f :=
  rfl


@[ext]
theorem ext {f g : C₀(α, β)} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


@[simp]
lemma coe_mk {f : α → β} (hf : Continuous f) (hf' : Tendsto f (cocompact α) (𝓝 0)) :
    { toFun := f,
      continuous_toFun := hf,
      zero_at_infty' := hf' : ZeroAtInftyContinuousMap α β} = f :=
  rfl


/-- Copy of a `ZeroAtInftyContinuousMap` with a new `toFun` equal to the old one. Useful
to fix definitional equalities. -/
protected def copy (f : C₀(α, β)) (f' : α → β) (h : f' = f) : C₀(α, β) where
  toFun := f'
  continuous_toFun := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      f : ZeroAtInftyContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Continuous f'
    -/
    rw [h]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      f : ZeroAtInftyContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Continuous ⇑f
    -/
    exact f.continuous_toFun
    /-
      🎉 no goals
    -/
  zero_at_infty' := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      f : ZeroAtInftyContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Filter.Tendsto { toFun := f', continuous_toFun := ⋯ }.toFun (Filter.cocompac …
    -/
    simp_rw [h]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : Zero β
      inst✝¹ : FunLike F α β
      inst✝ : ZeroAtInftyContinuousMapClass F α β
      f : ZeroAtInftyContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Filter.Tendsto (⇑f) (Filter.cocompact α) (nhds 0)
    -/
    exact f.zero_at_infty'
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_copy (f : C₀(α, β)) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : C₀(α, β)) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


theorem eq_of_empty [IsEmpty α] (f g : C₀(α, β)) : f = g :=
  ext <| IsEmpty.elim ‹_›


/-- A continuous function on a compact space is automatically a continuous function vanishing at
infinity. -/
@[simps]
def ContinuousMap.liftZeroAtInfty [CompactSpace α] : C(α, β) ≃ C₀(α, β) where
  toFun f :=
    { toFun := f
      continuous_toFun := f.continuous
                           /-
                             F : Type u_1
                             α : Type u
                             β : Type v
                             γ : Type w
                             inst✝⁵ : TopologicalSpace α
                             inst✝⁴ : TopologicalSpace β
                             inst✝³ : Zero β
                             inst✝² : FunLike F α β
                             inst✝¹ : ZeroAtInftyContinuousMapClass F α β
                             inst✝ : CompactSpace α
                             f : ContinuousMap α β
                             ⊢ Filter.Tendsto { toFun := ⇑f, continuous_toFun := ⋯ }.toFun (Filter.cocompac …
                           -/
      zero_at_infty' := by simp }
                           /-
                             🎉 no goals
                           -/
  invFun f := f
  left_inv f := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : TopologicalSpace β
      inst✝³ : Zero β
      inst✝² : FunLike F α β
      inst✝¹ : ZeroAtInftyContinuousMapClass F α β
      inst✝ : CompactSpace α
      f : ContinuousMap α β
      ⊢ Eq ((fun f => ↑f) ((fun f => { toFun := ⇑f, continuous_toFun := ⋯, zero_at_i …
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : TopologicalSpace β
      inst✝³ : Zero β
      inst✝² : FunLike F α β
      inst✝¹ : ZeroAtInftyContinuousMapClass F α β
      inst✝ : CompactSpace α
      f : ContinuousMap α β
      a✝ : α
      ⊢ Eq (((fun f => ↑f) ((fun f => { toFun := ⇑f, continuous_toFun := ⋯, zero_at_ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : TopologicalSpace β
      inst✝³ : Zero β
      inst✝² : FunLike F α β
      inst✝¹ : ZeroAtInftyContinuousMapClass F α β
      inst✝ : CompactSpace α
      f : ZeroAtInftyContinuousMap α β
      ⊢ Eq ((fun f => { toFun := ⇑f, continuous_toFun := ⋯, zero_at_infty' := ⋯ }) ( …
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : TopologicalSpace β
      inst✝³ : Zero β
      inst✝² : FunLike F α β
      inst✝¹ : ZeroAtInftyContinuousMapClass F α β
      inst✝ : CompactSpace α
      f : ZeroAtInftyContinuousMap α β
      x✝ : α
      ⊢ Eq (((fun f => { toFun := ⇑f, continuous_toFun := ⋯, zero_at_infty' := ⋯ })  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A continuous function on a compact space is automatically a continuous function vanishing at
infinity. This is not an instance to avoid type class loops. -/
lemma zeroAtInftyContinuousMapClass.ofCompact {G : Type*} [FunLike G α β]
    [ContinuousMapClass G α β] [CompactSpace α] : ZeroAtInftyContinuousMapClass G α β where
  map_continuous := map_continuous
                      /-
                        α : Type u
                        β : Type v
                        inst✝⁵ : TopologicalSpace α
                        inst✝⁴ : TopologicalSpace β
                        inst✝³ : Zero β
                        G : Type u_2
                        inst✝² : FunLike G α β
                        inst✝¹ : ContinuousMapClass G α β
                        inst✝ : CompactSpace α
                        ⊢ ∀ (f : G), Filter.Tendsto (⇑f) (Filter.cocompact α) (nhds 0)
                      -/
  zero_at_infty := by simp
                      /-
                        🎉 no goals
                      -/


instance instZero [Zero β] : Zero C₀(α, β) :=
  ⟨⟨0, tendsto_const_nhds⟩⟩


instance instInhabited [Zero β] : Inhabited C₀(α, β) :=
  ⟨0⟩


@[simp]
theorem coe_zero [Zero β] : ⇑(0 : C₀(α, β)) = 0 :=
  rfl


theorem zero_apply [Zero β] : (0 : C₀(α, β)) x = 0 :=
  rfl


instance instMul [MulZeroClass β] [ContinuousMul β] : Mul C₀(α, β) :=
  ⟨fun f g =>
               /-
                 F : Type u_1
                 α : Type u
                 β : Type v
                 γ : Type w
                 inst✝³ : TopologicalSpace α
                 inst✝² : TopologicalSpace β
                 x : α
                 inst✝¹ : MulZeroClass β
                 inst✝ : ContinuousMul β
                 f g : ZeroAtInftyContinuousMap α β
                 ⊢ Filter.Tendsto (HMul.hMul ↑f ↑g).toFun (Filter.cocompact α) (nhds 0)
               -/
    ⟨f * g, by simpa only [mul_zero] using (zero_at_infty f).mul (zero_at_infty g)⟩⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem coe_mul [MulZeroClass β] [ContinuousMul β] (f g : C₀(α, β)) : ⇑(f * g) = f * g :=
  rfl


theorem mul_apply [MulZeroClass β] [ContinuousMul β] (f g : C₀(α, β)) : (f * g) x = f x * g x :=
  rfl


instance instMulZeroClass [MulZeroClass β] [ContinuousMul β] : MulZeroClass C₀(α, β) :=
  DFunLike.coe_injective.mulZeroClass _ coe_zero coe_mul


instance instSemigroupWithZero [SemigroupWithZero β] [ContinuousMul β] :
    SemigroupWithZero C₀(α, β) :=
  DFunLike.coe_injective.semigroupWithZero _ coe_zero coe_mul


instance instAdd [AddZeroClass β] [ContinuousAdd β] : Add C₀(α, β) :=
                         /-
                           F : Type u_1
                           α : Type u
                           β : Type v
                           γ : Type w
                           inst✝³ : TopologicalSpace α
                           inst✝² : TopologicalSpace β
                           x : α
                           inst✝¹ : AddZeroClass β
                           inst✝ : ContinuousAdd β
                           f g : ZeroAtInftyContinuousMap α β
                           ⊢ Filter.Tendsto (HAdd.hAdd ↑f ↑g).toFun (Filter.cocompact α) (nhds 0)
                         -/
  ⟨fun f g => ⟨f + g, by simpa only [add_zero] using (zero_at_infty f).add (zero_at_infty g)⟩⟩
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem coe_add [AddZeroClass β] [ContinuousAdd β] (f g : C₀(α, β)) : ⇑(f + g) = f + g :=
  rfl


theorem add_apply [AddZeroClass β] [ContinuousAdd β] (f g : C₀(α, β)) : (f + g) x = f x + g x :=
  rfl


instance instAddZeroClass [AddZeroClass β] [ContinuousAdd β] : AddZeroClass C₀(α, β) :=
  DFunLike.coe_injective.addZeroClass _ coe_zero coe_add


instance instSMul [Zero β] {R : Type*} [Zero R] [SMulWithZero R β] [ContinuousConstSMul R β] :
    SMul R C₀(α, β) :=
  -- Porting note: Original version didn't have `Continuous.const_smul f.continuous r`
  ⟨fun r f => ⟨⟨r • ⇑f, Continuous.const_smul f.continuous r⟩,
       /-
         F : Type u_1
         α : Type u
         β : Type v
         γ : Type w
         inst✝⁵ : TopologicalSpace α
         inst✝⁴ : TopologicalSpace β
         x : α
         inst✝³ : Zero β
         R : Type u_2
         inst✝² : Zero R
         inst✝¹ : SMulWithZero R β
         inst✝ : ContinuousConstSMul R β
         r : R
         f : ZeroAtInftyContinuousMap α β
         ⊢ Filter.Tendsto { toFun := HSMul.hSMul r ⇑f, continuous_toFun := ⋯ }.toFun (F …
       -/
    by simpa [smul_zero] using (zero_at_infty f).const_smul r⟩⟩
       /-
         🎉 no goals
       -/


@[simp, norm_cast]
theorem coe_smul [Zero β] {R : Type*} [Zero R] [SMulWithZero R β] [ContinuousConstSMul R β] (r : R)
    (f : C₀(α, β)) : ⇑(r • f) = r • ⇑f :=
  rfl


theorem smul_apply [Zero β] {R : Type*} [Zero R] [SMulWithZero R β] [ContinuousConstSMul R β]
    (r : R) (f : C₀(α, β)) (x : α) : (r • f) x = r • f x :=
  rfl


instance instAddMonoid : AddMonoid C₀(α, β) :=
  DFunLike.coe_injective.addMonoid _ coe_zero coe_add fun _ _ => rfl


instance instAddCommMonoid [AddCommMonoid β] [ContinuousAdd β] : AddCommMonoid C₀(α, β) :=
  DFunLike.coe_injective.addCommMonoid _ coe_zero coe_add fun _ _ => rfl


instance instNeg : Neg C₀(α, β) :=
                    /-
                      F : Type u_1
                      α : Type u
                      β : Type v
                      γ : Type w
                      inst✝³ : TopologicalSpace α
                      inst✝² : TopologicalSpace β
                      x : α
                      inst✝¹ : AddGroup β
                      inst✝ : TopologicalAddGroup β
                      f✝ g f : ZeroAtInftyContinuousMap α β
                      ⊢ Filter.Tendsto (Neg.neg ↑f).toFun (Filter.cocompact α) (nhds 0)
                    -/
  ⟨fun f => ⟨-f, by simpa only [neg_zero] using (zero_at_infty f).neg⟩⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem coe_neg : ⇑(-f) = -f :=
  rfl


theorem neg_apply : (-f) x = -f x :=
  rfl


instance instSub : Sub C₀(α, β) :=
                         /-
                           F : Type u_1
                           α : Type u
                           β : Type v
                           γ : Type w
                           inst✝³ : TopologicalSpace α
                           inst✝² : TopologicalSpace β
                           x : α
                           inst✝¹ : AddGroup β
                           inst✝ : TopologicalAddGroup β
                           f✝ g✝ f g : ZeroAtInftyContinuousMap α β
                           ⊢ Filter.Tendsto (HSub.hSub ↑f ↑g).toFun (Filter.cocompact α) (nhds 0)
                         -/
  ⟨fun f g => ⟨f - g, by simpa only [sub_zero] using (zero_at_infty f).sub (zero_at_infty g)⟩⟩
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem coe_sub : ⇑(f - g) = f - g :=
  rfl


theorem sub_apply : (f - g) x = f x - g x :=
  rfl


instance instAddGroup : AddGroup C₀(α, β) :=
  DFunLike.coe_injective.addGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ => rfl


instance instAddCommGroup [AddCommGroup β] [TopologicalAddGroup β] : AddCommGroup C₀(α, β) :=
  DFunLike.coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ =>
    rfl


instance instIsCentralScalar [Zero β] {R : Type*} [Zero R] [SMulWithZero R β] [SMulWithZero Rᵐᵒᵖ β]
    [ContinuousConstSMul R β] [IsCentralScalar R β] : IsCentralScalar R C₀(α, β) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


instance instSMulWithZero [Zero β] {R : Type*} [Zero R] [SMulWithZero R β]
    [ContinuousConstSMul R β] : SMulWithZero R C₀(α, β) :=
  Function.Injective.smulWithZero ⟨_, coe_zero⟩ DFunLike.coe_injective coe_smul


instance instMulActionWithZero [Zero β] {R : Type*} [MonoidWithZero R] [MulActionWithZero R β]
    [ContinuousConstSMul R β] : MulActionWithZero R C₀(α, β) :=
  Function.Injective.mulActionWithZero ⟨_, coe_zero⟩ DFunLike.coe_injective coe_smul


instance instModule [AddCommMonoid β] [ContinuousAdd β] {R : Type*} [Semiring R] [Module R β]
    [ContinuousConstSMul R β] : Module R C₀(α, β) :=
  Function.Injective.module R ⟨⟨_, coe_zero⟩, coe_add⟩ DFunLike.coe_injective coe_smul


instance instNonUnitalNonAssocSemiring [NonUnitalNonAssocSemiring β] [TopologicalSemiring β] :
    NonUnitalNonAssocSemiring C₀(α, β) :=
  DFunLike.coe_injective.nonUnitalNonAssocSemiring _ coe_zero coe_add coe_mul fun _ _ => rfl


instance instNonUnitalSemiring [NonUnitalSemiring β] [TopologicalSemiring β] :
    NonUnitalSemiring C₀(α, β) :=
  DFunLike.coe_injective.nonUnitalSemiring _ coe_zero coe_add coe_mul fun _ _ => rfl


instance instNonUnitalCommSemiring [NonUnitalCommSemiring β] [TopologicalSemiring β] :
    NonUnitalCommSemiring C₀(α, β) :=
  DFunLike.coe_injective.nonUnitalCommSemiring _ coe_zero coe_add coe_mul fun _ _ => rfl


instance instNonUnitalNonAssocRing [NonUnitalNonAssocRing β] [TopologicalRing β] :
    NonUnitalNonAssocRing C₀(α, β) :=
  DFunLike.coe_injective.nonUnitalNonAssocRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ => rfl) fun _ _ => rfl


instance instNonUnitalRing [NonUnitalRing β] [TopologicalRing β] : NonUnitalRing C₀(α, β) :=
  DFunLike.coe_injective.nonUnitalRing _ coe_zero coe_add coe_mul coe_neg coe_sub (fun _ _ => rfl)
    fun _ _ => rfl


instance instNonUnitalCommRing [NonUnitalCommRing β] [TopologicalRing β] :
    NonUnitalCommRing C₀(α, β) :=
  DFunLike.coe_injective.nonUnitalCommRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ => rfl) fun _ _ => rfl


instance instIsScalarTower {R : Type*} [Semiring R] [NonUnitalNonAssocSemiring β]
    [TopologicalSemiring β] [Module R β] [ContinuousConstSMul R β] [IsScalarTower R β β] :
    IsScalarTower R C₀(α, β) C₀(α, β) where
  smul_assoc r f g := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_2
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : IsScalarTower R β β
      r : R
      f g : ZeroAtInftyContinuousMap α β
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r f) g) (HSMul.hSMul r (HSMul.hSMul f g))
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_2
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : IsScalarTower R β β
      r : R
      f g : ZeroAtInftyContinuousMap α β
      x✝ : α
      ⊢ Eq ((HSMul.hSMul (HSMul.hSMul r f) g) x✝) ((HSMul.hSMul r (HSMul.hSMul f g)) …
    -/
    simp only [smul_eq_mul, coe_mul, coe_smul, Pi.mul_apply, Pi.smul_apply]
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_2
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : IsScalarTower R β β
      r : R
      f g : ZeroAtInftyContinuousMap α β
      x✝ : α
      ⊢ Eq (HMul.hMul (HSMul.hSMul r (f x✝)) (g x✝)) (HSMul.hSMul r (HMul.hMul (f x✝ …
    -/
    rw [← smul_eq_mul, ← smul_eq_mul, smul_assoc]
    /-
      🎉 no goals
    -/


instance instSMulCommClass {R : Type*} [Semiring R] [NonUnitalNonAssocSemiring β]
    [TopologicalSemiring β] [Module R β] [ContinuousConstSMul R β] [SMulCommClass R β β] :
    SMulCommClass R C₀(α, β) C₀(α, β) where
  smul_comm r f g := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_2
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : SMulCommClass R β β
      r : R
      f g : ZeroAtInftyContinuousMap α β
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul f g)) (HSMul.hSMul f (HSMul.hSMul r g))
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_2
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : SMulCommClass R β β
      r : R
      f g : ZeroAtInftyContinuousMap α β
      x✝ : α
      ⊢ Eq ((HSMul.hSMul r (HSMul.hSMul f g)) x✝) ((HSMul.hSMul f (HSMul.hSMul r g)) …
    -/
    simp only [smul_eq_mul, coe_smul, coe_mul, Pi.smul_apply, Pi.mul_apply]
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_2
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : SMulCommClass R β β
      r : R
      f g : ZeroAtInftyContinuousMap α β
      x✝ : α
      ⊢ Eq (HSMul.hSMul r (HMul.hMul (f x✝) (g x✝))) (HMul.hMul (f x✝) (HSMul.hSMul  …
    -/
    rw [← smul_eq_mul, ← smul_eq_mul, smul_comm]
    /-
      🎉 no goals
    -/


theorem uniformContinuous (f : F) : UniformContinuous (f : β → γ) :=
  (map_continuous f).uniformContinuous_of_tendsto_cocompact (zero_at_infty f)


protected theorem bounded (f : F) : ∃ C, ∀ x y : α, dist ((f : α → β) x) (f y) ≤ C := by
  obtain ⟨K : Set α, hK₁, hK₂⟩ := mem_cocompact.mp
    (tendsto_def.mp (zero_at_infty (f : F)) _ (closedBall_mem_nhds (0 : β) zero_lt_one))
  /-
    case intro.intro
    F : Type u_1
    α : Type u
    β : Type v
    inst✝⁴ : TopologicalSpace α
    inst✝³ : PseudoMetricSpace β
    inst✝² : Zero β
    inst✝¹ : FunLike F α β
    inst✝ : ZeroAtInftyContinuousMapClass F α β
    f : F
    K : Set α
    hK₁ : IsCompact K
    hK₂ : HasSubset.Subset (HasCompl.compl K) (Set.preimage (⇑f) (Metric.closedBal …
    ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C
  -/
  obtain ⟨C, hC⟩ := (hK₁.image (map_continuous f)).isBounded.subset_closedBall (0 : β)
  /-
    case intro.intro.intro
    F : Type u_1
    α : Type u
    β : Type v
    inst✝⁴ : TopologicalSpace α
    inst✝³ : PseudoMetricSpace β
    inst✝² : Zero β
    inst✝¹ : FunLike F α β
    inst✝ : ZeroAtInftyContinuousMapClass F α β
    f : F
    K : Set α
    hK₁ : IsCompact K
    hK₂ : HasSubset.Subset (HasCompl.compl K) (Set.preimage (⇑f) (Metric.closedBal …
    C : Real
    hC : HasSubset.Subset (Set.image (⇑f) K) (Metric.closedBall 0 C)
    ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C
  -/
  refine ⟨max C 1 + max C 1, fun x y => ?_⟩
  have : ∀ x, f x ∈ closedBall (0 : β) (max C 1) := by
    intro x
    by_cases hx : x ∈ K
    · exact (mem_closedBall.mp <| hC ⟨x, hx, rfl⟩).trans (le_max_left _ _)
    · exact (mem_closedBall.mp <| mem_preimage.mp (hK₂ hx)).trans (le_max_right _ _)
  exact (dist_triangle (f x) 0 (f y)).trans
    (add_le_add (mem_closedBall.mp <| this x) (mem_closedBall'.mp <| this y))


theorem isBounded_range (f : C₀(α, β)) : IsBounded (range f) :=
  isBounded_range_iff.2 (ZeroAtInftyContinuousMap.bounded f)


theorem isBounded_image (f : C₀(α, β)) (s : Set α) : IsBounded (f '' s) :=
  f.isBounded_range.subset <| image_subset_range _ _


instance (priority := 100) instBoundedContinuousMapClass : BoundedContinuousMapClass F α β :=
  { ‹ZeroAtInftyContinuousMapClass F α β› with
    map_bounded := fun f => ZeroAtInftyContinuousMap.bounded f }


/-- Construct a bounded continuous function from a continuous function vanishing at infinity. -/
@[simps!]
def toBCF (f : C₀(α, β)) : α →ᵇ β :=
  ⟨f, map_bounded f⟩


theorem toBCF_injective : Function.Injective (toBCF : C₀(α, β) → α →ᵇ β) := fun f g h => by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : Zero β
    f g : ZeroAtInftyContinuousMap α β
    h : Eq f.toBCF g.toBCF
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : Zero β
    f g : ZeroAtInftyContinuousMap α β
    h : Eq f.toBCF g.toBCF
    x : α
    ⊢ Eq (f x) (g x)
  -/
  simpa only using DFunLike.congr_fun h x
  /-
    🎉 no goals
  -/


/-- The type of continuous functions vanishing at infinity, with the uniform distance induced by the
inclusion `ZeroAtInftyContinuousMap.toBCF`, is a pseudo-metric space. -/
noncomputable instance instPseudoMetricSpace : PseudoMetricSpace C₀(α, β) :=
  PseudoMetricSpace.induced toBCF inferInstance


/-- The type of continuous functions vanishing at infinity, with the uniform distance induced by the
inclusion `ZeroAtInftyContinuousMap.toBCF`, is a metric space. -/
noncomputable instance instMetricSpace {β : Type*} [MetricSpace β] [Zero β] :
    MetricSpace C₀(α, β) :=
  MetricSpace.induced _ (toBCF_injective α β) inferInstance


@[simp]
theorem dist_toBCF_eq_dist {f g : C₀(α, β)} : dist f.toBCF g.toBCF = dist f g :=
  rfl


/-- Convergence in the metric on `C₀(α, β)` is uniform convergence. -/
theorem tendsto_iff_tendstoUniformly {ι : Type*} {F : ι → C₀(α, β)} {f : C₀(α, β)} {l : Filter ι} :
    Tendsto F l (𝓝 f) ↔ TendstoUniformly (fun i => F i) f l := by
  simpa only [Metric.tendsto_nhds] using
    @BoundedContinuousFunction.tendsto_iff_tendstoUniformly _ _ _ _ _ (fun i => (F i).toBCF)
      f.toBCF l


                                                                    /-
                                                                      α : Type u
                                                                      β : Type v
                                                                      inst✝² : TopologicalSpace α
                                                                      inst✝¹ : PseudoMetricSpace β
                                                                      inst✝ : Zero β
                                                                      ⊢ Isometry ZeroAtInftyContinuousMap.toBCF
                                                                    -/
theorem isometry_toBCF : Isometry (toBCF : C₀(α, β) → α →ᵇ β) := by tauto
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem isClosed_range_toBCF : IsClosed (range (toBCF : C₀(α, β) → α →ᵇ β)) := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : Zero β
    ⊢ IsClosed (Set.range ZeroAtInftyContinuousMap.toBCF)
  -/
  refine isClosed_iff_clusterPt.mpr fun f hf => ?_
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : Zero β
    f : BoundedContinuousFunction α β
    hf : ClusterPt f (Filter.principal (Set.range ZeroAtInftyContinuousMap.toBCF))
    ⊢ Membership.mem (Set.range ZeroAtInftyContinuousMap.toBCF) f
  -/
  rw [clusterPt_principal_iff] at hf
  have : Tendsto f (cocompact α) (𝓝 0) := by
    refine Metric.tendsto_nhds.mpr fun ε hε => ?_
    obtain ⟨_, hg, g, rfl⟩ := hf (ball f (ε / 2)) (ball_mem_nhds f <| half_pos hε)
    refine (Metric.tendsto_nhds.mp (zero_at_infty g) (ε / 2) (half_pos hε)).mp
      (Eventually.of_forall fun x hx => ?_)
    calc
      dist (f x) 0 ≤ dist (g.toBCF x) (f x) + dist (g x) 0 := dist_triangle_left _ _ _
      _ < dist g.toBCF f + ε / 2 := add_lt_add_of_le_of_lt (dist_coe_le_dist x) hx
      _ < ε := by simpa [add_halves ε] using add_lt_add_right (mem_ball.1 hg) (ε / 2)
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : Zero β
    f : BoundedContinuousFunction α β
    hf : ∀ (U : Set (BoundedContinuousFunction α β)), Membership.mem (nhds f) U →  …
    this : Filter.Tendsto (⇑f) (Filter.cocompact α) (nhds 0)
    ⊢ Membership.mem (Set.range ZeroAtInftyContinuousMap.toBCF) f
  -/
  exact ⟨⟨f.toContinuousMap, this⟩, rfl⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-17")] alias closed_range_toBCF := isClosed_range_toBCF


/-- Continuous functions vanishing at infinity taking values in a complete space form a
complete space. -/
instance instCompleteSpace [CompleteSpace β] : CompleteSpace C₀(α, β) :=
  (completeSpace_iff_isComplete_range isometry_toBCF.isUniformInducing).mpr
    isClosed_range_toBCF.isComplete


noncomputable instance instSeminormedAddCommGroup [SeminormedAddCommGroup β] :
    SeminormedAddCommGroup C₀(α, β) :=
  SeminormedAddCommGroup.induced _ _ (⟨⟨toBCF, rfl⟩, fun _ _ => rfl⟩ : C₀(α, β) →+ α →ᵇ β)


noncomputable instance instNormedAddCommGroup [NormedAddCommGroup β] :
    NormedAddCommGroup C₀(α, β) :=
  NormedAddCommGroup.induced _ _ (⟨⟨toBCF, rfl⟩, fun _ _ => rfl⟩ : C₀(α, β) →+ α →ᵇ β)
    (toBCF_injective α β)


@[simp]
theorem norm_toBCF_eq_norm {f : C₀(α, β)} : ‖f.toBCF‖ = ‖f‖ :=
  rfl


instance : NormedSpace 𝕜 C₀(α, β) where norm_smul_le k f := (norm_smul_le k f.toBCF : _)


noncomputable instance instNonUnitalSeminormedRing [NonUnitalSeminormedRing β] :
    NonUnitalSeminormedRing C₀(α, β) :=
  { instNonUnitalRing, instSeminormedAddCommGroup with
    norm_mul := fun f g => norm_mul_le f.toBCF g.toBCF }


noncomputable instance instNonUnitalNormedRing [NonUnitalNormedRing β] :
    NonUnitalNormedRing C₀(α, β) :=
  { instNonUnitalRing, instNormedAddCommGroup with
    norm_mul := fun f g => norm_mul_le f.toBCF g.toBCF }


noncomputable instance instNonUnitalSeminormedCommRing [NonUnitalSeminormedCommRing β] :
    NonUnitalSeminormedCommRing C₀(α, β) :=
  { instNonUnitalSeminormedRing, instNonUnitalCommRing with }


noncomputable instance instNonUnitalNormedCommRing [NonUnitalNormedCommRing β] :
    NonUnitalNormedCommRing C₀(α, β) :=
  { instNonUnitalNormedRing, instNonUnitalCommRing with }


instance instStar : Star C₀(α, β) where
  star f :=
    { toFun := fun x => star (f x)
      continuous_toFun := (map_continuous f).star
      zero_at_infty' := by
        /-
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝⁴ : TopologicalSpace α
          inst✝³ : TopologicalSpace β
          inst✝² : AddMonoid β
          inst✝¹ : StarAddMonoid β
          inst✝ : ContinuousStar β
          f : ZeroAtInftyContinuousMap α β
          ⊢ Filter.Tendsto { toFun := fun x => Star.star (f x), continuous_toFun := ⋯ }. …
        -/
        simpa only [star_zero] using (continuous_star.tendsto (0 : β)).comp (zero_at_infty f) }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_star (f : C₀(α, β)) : ⇑(star f) = star (⇑f) :=
  rfl


theorem star_apply (f : C₀(α, β)) (x : α) : (star f) x = star (f x) :=
  rfl


instance instStarAddMonoid [ContinuousAdd β] : StarAddMonoid C₀(α, β) where
  star_involutive f := ext fun x => star_star (f x)
  star_add f g := ext fun x => star_add (f x) (g x)


instance instNormedStarGroup : NormedStarGroup C₀(α, β) where
  norm_star f := (norm_star f.toBCF : _)


instance instStarModule : StarModule 𝕜 C₀(α, β) where
  star_smul k f := ext fun x => star_smul k (f x)


instance instStarRing : StarRing C₀(α, β) :=
  { ZeroAtInftyContinuousMap.instStarAddMonoid with
    star_mul := fun f g => ext fun x => star_mul (f x) (g x) }


instance instCStarRing [NonUnitalNormedRing β] [StarRing β] [CStarRing β] : CStarRing C₀(α, β) where
  norm_mul_self_le f := CStarRing.norm_mul_self_le (x := f.toBCF)


local notation α " →co " β => CocompactMap α β


/-- Composition of a continuous function vanishing at infinity with a cocompact map yields another
continuous function vanishing at infinity. -/
def comp (f : C₀(γ, δ)) (g : β →co γ) : C₀(β, δ) where
  toContinuousMap := (f : C(γ, δ)).comp g
  zero_at_infty' := (zero_at_infty f).comp (cocompact_tendsto g)


@[simp]
theorem coe_comp_to_continuous_fun (f : C₀(γ, δ)) (g : β →co γ) : ((f.comp g) : β → δ) = f ∘ g :=
  rfl


@[simp]
theorem comp_id (f : C₀(γ, δ)) : f.comp (CocompactMap.id γ) = f :=
  ext fun _ => rfl


@[simp]
theorem comp_assoc (f : C₀(γ, δ)) (g : β →co γ) (h : α →co β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem zero_comp (g : β →co γ) : (0 : C₀(γ, δ)).comp g = 0 :=
  rfl


/-- Composition as an additive monoid homomorphism. -/
def compAddMonoidHom [AddMonoid δ] [ContinuousAdd δ] (g : β →co γ) : C₀(γ, δ) →+ C₀(β, δ) where
  toFun f := f.comp g
  map_zero' := zero_comp g
  map_add' _ _ := rfl


/-- Composition as a semigroup homomorphism. -/
def compMulHom [MulZeroClass δ] [ContinuousMul δ] (g : β →co γ) : C₀(γ, δ) →ₙ* C₀(β, δ) where
  toFun f := f.comp g
  map_mul' _ _ := rfl


/-- Composition as a linear map. -/
def compLinearMap [AddCommMonoid δ] [ContinuousAdd δ] {R : Type*} [Semiring R] [Module R δ]
    [ContinuousConstSMul R δ] (g : β →co γ) : C₀(γ, δ) →ₗ[R] C₀(β, δ) where
  toFun f := f.comp g
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- Composition as a non-unital algebra homomorphism. -/
def compNonUnitalAlgHom {R : Type*} [Semiring R] [NonUnitalNonAssocSemiring δ]
    [TopologicalSemiring δ] [Module R δ] [ContinuousConstSMul R δ] (g : β →co γ) :
    C₀(γ, δ) →ₙₐ[R] C₀(β, δ) where
  toFun f := f.comp g
  map_smul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl


