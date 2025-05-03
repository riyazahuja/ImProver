/-- The type of centroid homomorphisms from `α` to `α`. -/
structure CentroidHom (α : Type*) [NonUnitalNonAssocSemiring α] extends α →+ α where
  /-- Commutativity of centroid homomorphims with left multiplication. -/
  map_mul_left' (a b : α) : toFun (a * b) = a * toFun b
  /-- Commutativity of centroid homomorphims with right multiplication. -/
  map_mul_right' (a b : α) : toFun (a * b) = toFun a * b


/-- `CentroidHomClass F α` states that `F` is a type of centroid homomorphisms.

You should extend this class when you extend `CentroidHom`. -/
class CentroidHomClass (F : Type*) (α : outParam Type*)
    [NonUnitalNonAssocSemiring α] [FunLike F α α] extends AddMonoidHomClass F α α : Prop where
  /-- Commutativity of centroid homomorphims with left multiplication. -/
  map_mul_left (f : F) (a b : α) : f (a * b) = a * f b
  /-- Commutativity of centroid homomorphims with right multiplication. -/
  map_mul_right (f : F) (a b : α) : f (a * b) = f a * b



instance [NonUnitalNonAssocSemiring α] [FunLike F α α] [CentroidHomClass F α] :
    CoeTC F (CentroidHom α) :=
  ⟨fun f ↦
    { (f : α →+ α) with
      toFun := f
      map_mul_left' := map_mul_left f
      map_mul_right' := map_mul_right f }⟩


instance : FunLike (CentroidHom α) α α where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R : Type u_4
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      f g : CentroidHom α
      h : Eq ((fun f => (↑f.toAddMonoidHom).toFun) f) ((fun f => (↑f.toAddMonoidHom) …
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R : Type u_4
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      g : CentroidHom α
      toAddMonoidHom✝ : AddMonoidHom α α
      map_mul_left'✝ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝).toFun (HMul.hMul a b)) (H …
      map_mul_right'✝ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝).toFun (HMul.hMul a b)) ( …
      h : Eq ((fun f => (↑f.toAddMonoidHom).toFun) { toAddMonoidHom := toAddMonoidHo …
      ⊢ Eq { toAddMonoidHom := toAddMonoidHom✝, map_mul_left' := map_mul_left'✝, map …
    -/
    cases g
    /-
      case mk.mk
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R : Type u_4
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      toAddMonoidHom✝¹ : AddMonoidHom α α
      map_mul_left'✝¹ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝¹).toFun (HMul.hMul a b))  …
      map_mul_right'✝¹ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝¹).toFun (HMul.hMul a b)) …
      toAddMonoidHom✝ : AddMonoidHom α α
      map_mul_left'✝ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝).toFun (HMul.hMul a b)) (H …
      map_mul_right'✝ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝).toFun (HMul.hMul a b)) ( …
      h : Eq ((fun f => (↑f.toAddMonoidHom).toFun) { toAddMonoidHom := toAddMonoidHo …
      ⊢ Eq { toAddMonoidHom := toAddMonoidHom✝¹, map_mul_left' := map_mul_left'✝¹, m …
    -/
    congr with x
    /-
      case mk.mk.e_toAddMonoidHom.h
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R : Type u_4
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      toAddMonoidHom✝¹ : AddMonoidHom α α
      map_mul_left'✝¹ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝¹).toFun (HMul.hMul a b))  …
      map_mul_right'✝¹ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝¹).toFun (HMul.hMul a b)) …
      toAddMonoidHom✝ : AddMonoidHom α α
      map_mul_left'✝ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝).toFun (HMul.hMul a b)) (H …
      map_mul_right'✝ : ∀ (a b : α), Eq ((↑toAddMonoidHom✝).toFun (HMul.hMul a b)) ( …
      h : Eq ((fun f => (↑f.toAddMonoidHom).toFun) { toAddMonoidHom := toAddMonoidHo …
      x : α
      ⊢ Eq (toAddMonoidHom✝¹ x) (toAddMonoidHom✝ x)
    -/
    exact congrFun h x
    /-
      🎉 no goals
    -/


instance : CentroidHomClass (CentroidHom α) α where
  map_zero f := f.map_zero'
  map_add f := f.map_add'
  map_mul_left f := f.map_mul_left'
  map_mul_right f := f.map_mul_right'


-- Porting note: removed @[simp]; not in normal form. (`toAddMonoidHom_eq_coe` below ensures that
-- the LHS simplifies to the RHS anyway.)

theorem toFun_eq_coe {f : CentroidHom α} : f.toFun = f := rfl


@[ext]
theorem ext {f g : CentroidHom α} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


@[simp, norm_cast]
theorem coe_toAddMonoidHom (f : CentroidHom α) : ⇑(f : α →+ α) = f :=
  rfl


@[simp]
theorem toAddMonoidHom_eq_coe (f : CentroidHom α) : f.toAddMonoidHom = f :=
  rfl


theorem coe_toAddMonoidHom_injective : Injective ((↑) : CentroidHom α → α →+ α) :=
  fun _f _g h => ext fun a ↦
    haveI := DFunLike.congr_fun h a
    this


/-- Turn a centroid homomorphism into an additive monoid endomorphism. -/
def toEnd (f : CentroidHom α) : AddMonoid.End α :=
  (f : α →+ α)


theorem toEnd_injective : Injective (CentroidHom.toEnd : CentroidHom α → AddMonoid.End α) :=
  coe_toAddMonoidHom_injective


/-- Copy of a `CentroidHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : CentroidHom α) (f' : α → α) (h : f' = f) : CentroidHom α :=
  { f.toAddMonoidHom.copy f' <| h with
    toFun := f'
                                  /-
                                    F : Type u_1
                                    M : Type u_2
                                    N : Type u_3
                                    R : Type u_4
                                    α : Type u_5
                                    inst✝ : NonUnitalNonAssocSemiring α
                                    f : CentroidHom α
                                    f' : α → α
                                    h : Eq f' ⇑f
                                    a b : α
                                    ⊢ Eq ((↑{ toFun := f', map_zero' := ⋯, map_add' := ⋯ }).toFun (HMul.hMul a b)) …
                                  -/
    map_mul_left' := fun a b ↦ by simp_rw [h, map_mul_left]
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     F : Type u_1
                                     M : Type u_2
                                     N : Type u_3
                                     R : Type u_4
                                     α : Type u_5
                                     inst✝ : NonUnitalNonAssocSemiring α
                                     f : CentroidHom α
                                     f' : α → α
                                     h : Eq f' ⇑f
                                     a b : α
                                     ⊢ Eq ((↑{ toFun := f', map_zero' := ⋯, map_add' := ⋯ }).toFun (HMul.hMul a b)) …
                                   -/
    map_mul_right' := fun a b ↦ by simp_rw [h, map_mul_right] }
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem coe_copy (f : CentroidHom α) (f' : α → α) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : CentroidHom α) (f' : α → α) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `CentroidHom`. -/
protected def id : CentroidHom α :=
  { AddMonoidHom.id α with
    map_mul_left' := fun _ _ ↦ rfl
    map_mul_right' := fun _ _ ↦ rfl }


instance : Inhabited (CentroidHom α) :=
  ⟨CentroidHom.id α⟩


@[simp, norm_cast]
theorem coe_id : ⇑(CentroidHom.id α) = id :=
  rfl


@[simp, norm_cast]
theorem toAddMonoidHom_id : (CentroidHom.id α : α →+ α) = AddMonoidHom.id α :=
  rfl


@[simp]
theorem id_apply (a : α) : CentroidHom.id α a = a :=
  rfl


/-- Composition of `CentroidHom`s as a `CentroidHom`. -/
def comp (g f : CentroidHom α) : CentroidHom α :=
  { g.toAddMonoidHom.comp f.toAddMonoidHom with
    map_mul_left' := fun _a _b ↦ (congr_arg g <| f.map_mul_left' _ _).trans <| g.map_mul_left' _ _
    map_mul_right' := fun _a _b ↦
      (congr_arg g <| f.map_mul_right' _ _).trans <| g.map_mul_right' _ _ }


@[simp, norm_cast]
theorem coe_comp (g f : CentroidHom α) : ⇑(g.comp f) = g ∘ f :=
  rfl


@[simp]
theorem comp_apply (g f : CentroidHom α) (a : α) : g.comp f a = g (f a) :=
  rfl


@[simp, norm_cast]
theorem coe_comp_addMonoidHom (g f : CentroidHom α) : (g.comp f : α →+ α) = (g : α →+ α).comp f :=
  rfl


@[simp]
theorem comp_assoc (h g f : CentroidHom α) : (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


@[simp]
theorem comp_id (f : CentroidHom α) : f.comp (CentroidHom.id α) = f :=
  rfl


@[simp]
theorem id_comp (f : CentroidHom α) : (CentroidHom.id α).comp f = f :=
  rfl


@[simp]
theorem cancel_right {g₁ g₂ f : CentroidHom α} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h ↦ ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun a ↦ congrFun (congrArg comp a) f⟩


@[simp]
theorem cancel_left {g f₁ f₂ : CentroidHom α} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                /-
                                  α : Type u_5
                                  inst✝ : NonUnitalNonAssocSemiring α
                                  g f₁ f₂ : CentroidHom α
                                  hg : Function.Injective ⇑g
                                  h : Eq (g.comp f₁) (g.comp f₂)
                                  a : α
                                  ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                -/
  ⟨fun h ↦ ext fun a ↦ hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                /-
                                  🎉 no goals
                                -/


instance : Zero (CentroidHom α) :=
  ⟨{ (0 : α →+ α) with
      map_mul_left' := fun _a _b ↦ (mul_zero _).symm
      map_mul_right' := fun _a _b ↦ (zero_mul _).symm }⟩


instance : One (CentroidHom α) :=
  ⟨CentroidHom.id α⟩


instance : Add (CentroidHom α) :=
  ⟨fun f g ↦
    { (f + g : α →+ α) with
      map_mul_left' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocSemiring α
          f g : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul a ((↑__src✝).toFun b))
        -/
        show f (a * b) + g (a * b) = a * (f b + g b)
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocSemiring α
          f g : CentroidHom α
          a b : α
          ⊢ Eq (HAdd.hAdd (f (HMul.hMul a b)) (g (HMul.hMul a b))) (HMul.hMul a (HAdd.hA …
        -/
        simp [map_mul_left, mul_add]
        /-
          🎉 no goals
        -/
      map_mul_right' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocSemiring α
          f g : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul ((↑__src✝).toFun a) b)
        -/
        show f (a * b) + g (a * b) = (f a + g a) * b
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocSemiring α
          f g : CentroidHom α
          a b : α
          ⊢ Eq (HAdd.hAdd (f (HMul.hMul a b)) (g (HMul.hMul a b))) (HMul.hMul (HAdd.hAdd …
        -/
        simp [map_mul_right, add_mul] }⟩
        /-
          🎉 no goals
        -/


instance : Mul (CentroidHom α) :=
  ⟨comp⟩


instance instSMul : SMul M (CentroidHom α) where
  smul n f :=
    { (n • f : α →+ α) with
      map_mul_left' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝¹² : NonUnitalNonAssocSemiring α
          inst✝¹¹ : Monoid M
          inst✝¹⁰ : Monoid N
          inst✝⁹ : Semiring R
          inst✝⁸ : DistribMulAction M α
          inst✝⁷ : SMulCommClass M α α
          inst✝⁶ : IsScalarTower M α α
          inst✝⁵ : DistribMulAction N α
          inst✝⁴ : SMulCommClass N α α
          inst✝³ : IsScalarTower N α α
          inst✝² : Module R α
          inst✝¹ : SMulCommClass R α α
          inst✝ : IsScalarTower R α α
          n : M
          f : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul a ((↑__src✝).toFun b))
        -/
        change n • f (a * b) = a * n • f b
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝¹² : NonUnitalNonAssocSemiring α
          inst✝¹¹ : Monoid M
          inst✝¹⁰ : Monoid N
          inst✝⁹ : Semiring R
          inst✝⁸ : DistribMulAction M α
          inst✝⁷ : SMulCommClass M α α
          inst✝⁶ : IsScalarTower M α α
          inst✝⁵ : DistribMulAction N α
          inst✝⁴ : SMulCommClass N α α
          inst✝³ : IsScalarTower N α α
          inst✝² : Module R α
          inst✝¹ : SMulCommClass R α α
          inst✝ : IsScalarTower R α α
          n : M
          f : CentroidHom α
          a b : α
          ⊢ Eq (HSMul.hSMul n (f (HMul.hMul a b))) (HMul.hMul a (HSMul.hSMul n (f b)))
        -/
        rw [map_mul_left f, ← mul_smul_comm]
        /-
          🎉 no goals
        -/
      map_mul_right' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝¹² : NonUnitalNonAssocSemiring α
          inst✝¹¹ : Monoid M
          inst✝¹⁰ : Monoid N
          inst✝⁹ : Semiring R
          inst✝⁸ : DistribMulAction M α
          inst✝⁷ : SMulCommClass M α α
          inst✝⁶ : IsScalarTower M α α
          inst✝⁵ : DistribMulAction N α
          inst✝⁴ : SMulCommClass N α α
          inst✝³ : IsScalarTower N α α
          inst✝² : Module R α
          inst✝¹ : SMulCommClass R α α
          inst✝ : IsScalarTower R α α
          n : M
          f : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul ((↑__src✝).toFun a) b)
        -/
        change n • f (a * b) = n • f a * b
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝¹² : NonUnitalNonAssocSemiring α
          inst✝¹¹ : Monoid M
          inst✝¹⁰ : Monoid N
          inst✝⁹ : Semiring R
          inst✝⁸ : DistribMulAction M α
          inst✝⁷ : SMulCommClass M α α
          inst✝⁶ : IsScalarTower M α α
          inst✝⁵ : DistribMulAction N α
          inst✝⁴ : SMulCommClass N α α
          inst✝³ : IsScalarTower N α α
          inst✝² : Module R α
          inst✝¹ : SMulCommClass R α α
          inst✝ : IsScalarTower R α α
          n : M
          f : CentroidHom α
          a b : α
          ⊢ Eq (HSMul.hSMul n (f (HMul.hMul a b))) (HMul.hMul (HSMul.hSMul n (f a)) b)
        -/
        rw [map_mul_right f, ← smul_mul_assoc] }
        /-
          🎉 no goals
        -/


instance [SMul M N] [IsScalarTower M N α] : IsScalarTower M N (CentroidHom α) where
  smul_assoc _ _ _ := ext fun _ => smul_assoc _ _ _


instance [SMulCommClass M N α] : SMulCommClass M N (CentroidHom α) where
  smul_comm _ _ _ := ext fun _ => smul_comm _ _ _


instance [DistribMulAction Mᵐᵒᵖ α] [IsCentralScalar M α] : IsCentralScalar M (CentroidHom α) where
  op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


instance isScalarTowerRight : IsScalarTower M (CentroidHom α) (CentroidHom α) where
  smul_assoc _ _ _ := rfl


instance hasNPowNat : Pow (CentroidHom α) ℕ :=
  ⟨fun f n ↦
    { toAddMonoidHom := (f.toEnd ^ n : AddMonoid.End α)
      map_mul_left' := fun a b ↦ by
        induction n with
        | zero => rfl
        | succ n ih =>
          rw [pow_succ']
          exact (congr_arg f.toEnd ih).trans (f.map_mul_left' _ _)
      map_mul_right' := fun a b ↦ by
        induction n with
        | zero => rfl
        | succ n ih =>
          rw [pow_succ']
          exact (congr_arg f.toEnd ih).trans (f.map_mul_right' _ _)}⟩


@[simp, norm_cast]
theorem coe_zero : ⇑(0 : CentroidHom α) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_one : ⇑(1 : CentroidHom α) = id :=
  rfl


@[simp, norm_cast]
theorem coe_add (f g : CentroidHom α) : ⇑(f + g) = f + g :=
  rfl


@[simp, norm_cast]
theorem coe_mul (f g : CentroidHom α) : ⇑(f * g) = f ∘ g :=
  rfl


@[simp, norm_cast]
theorem coe_smul (n : M) (f : CentroidHom α) : ⇑(n • f) = n • ⇑f :=
  rfl


@[simp]
theorem zero_apply (a : α) : (0 : CentroidHom α) a = 0 :=
  rfl


@[simp]
theorem one_apply (a : α) : (1 : CentroidHom α) a = a :=
  rfl


@[simp]
theorem add_apply (f g : CentroidHom α) (a : α) : (f + g) a = f a + g a :=
  rfl


@[simp]
theorem mul_apply (f g : CentroidHom α) (a : α) : (f * g) a = f (g a) :=
  rfl


@[simp]
theorem smul_apply (n : M) (f : CentroidHom α) (a : α) : (n • f) a = n • f a :=
  rfl


@[simp]
theorem toEnd_zero : (0 : CentroidHom α).toEnd = 0 :=
  rfl


@[simp]
theorem toEnd_add (x y : CentroidHom α) : (x + y).toEnd = x.toEnd + y.toEnd :=
  rfl


theorem toEnd_smul (m : M) (x : CentroidHom α) : (m • x).toEnd = m • x.toEnd :=
  rfl


instance : AddCommMonoid (CentroidHom α) :=
  coe_toAddMonoidHom_injective.addCommMonoid _ toEnd_zero toEnd_add (swap toEnd_smul)


instance : NatCast (CentroidHom α) where natCast n := n • (1 : CentroidHom α)


@[simp, norm_cast]
theorem coe_natCast (n : ℕ) : ⇑(n : CentroidHom α) = n • (CentroidHom.id α) :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := coe_natCast


theorem natCast_apply (n : ℕ) (m : α) : (n : CentroidHom α) m = n • m :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_apply := natCast_apply


@[simp]
theorem toEnd_one : (1 : CentroidHom α).toEnd = 1 :=
  rfl


@[simp]
theorem toEnd_mul (x y : CentroidHom α) : (x * y).toEnd = x.toEnd * y.toEnd :=
  rfl


@[simp]
theorem toEnd_pow (x : CentroidHom α) (n : ℕ) : (x ^ n).toEnd = x.toEnd ^ n :=
  rfl


@[simp, norm_cast]
theorem toEnd_natCast (n : ℕ) : (n : CentroidHom α).toEnd = ↑n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias toEnd_nat_cast := toEnd_natCast

-- cf `add_monoid.End.semiring`

instance : Semiring (CentroidHom α) :=
  toEnd_injective.semiring _ toEnd_zero toEnd_one toEnd_add toEnd_mul toEnd_smul toEnd_pow
    toEnd_natCast


variable (α) in
/-- `CentroidHom.toEnd` as a `RingHom`. -/
@[simps]
def toEndRingHom : CentroidHom α →+* AddMonoid.End α where
  toFun := toEnd
  map_zero' := toEnd_zero
  map_one' := toEnd_one
  map_add' := toEnd_add
  map_mul' := toEnd_mul


theorem comp_mul_comm (T S : CentroidHom α) (a b : α) : (T ∘ S) (a * b) = (S ∘ T) (a * b) := by
  /-
    α : Type u_5
    inst✝ : NonUnitalNonAssocSemiring α
    T S : CentroidHom α
    a b : α
    ⊢ Eq (Function.comp (⇑T) (⇑S) (HMul.hMul a b)) (Function.comp (⇑S) (⇑T) (HMul. …
  -/
  simp only [Function.comp_apply]
  /-
    α : Type u_5
    inst✝ : NonUnitalNonAssocSemiring α
    T S : CentroidHom α
    a b : α
    ⊢ Eq (T (S (HMul.hMul a b))) (S (T (HMul.hMul a b)))
  -/
  rw [map_mul_right, map_mul_left, ← map_mul_right, ← map_mul_left]
  /-
    🎉 no goals
  -/


instance : DistribMulAction M (CentroidHom α) :=
  toEnd_injective.distribMulAction (toEndRingHom α).toAddMonoidHom toEnd_smul


instance : Module R (CentroidHom α) :=
  toEnd_injective.module R (toEndRingHom α).toAddMonoidHom toEnd_smul


/-- The tautological action by `CentroidHom α` on `α`.

This generalizes `Function.End.applyMulAction`. -/
instance applyModule : Module (CentroidHom α) α where
  smul T a := T a
  add_smul _ _ _ := rfl
  zero_smul _ := rfl
  one_smul _ := rfl
  mul_smul _ _ _ := rfl
  smul_zero := map_zero
  smul_add := map_add


@[simp]
lemma smul_def (T : CentroidHom α) (a : α) : T • a = T a := rfl


instance : SMulCommClass (CentroidHom α) α α where
  smul_comm _ _ _ := map_mul_left _ _ _


instance : SMulCommClass α (CentroidHom α) α := SMulCommClass.symm _ _ _


instance : IsScalarTower (CentroidHom α) α α where
  smul_assoc _ _ _ := (map_mul_right _ _ _).symm


/-- The natural ring homomorphism from `R` into `CentroidHom α`.

This is a stronger version of `Module.toAddMonoidEnd`. -/
@[simps! apply_toFun]
def _root_.Module.toCentroidHom : R →+* CentroidHom α := RingHom.smulOneHom


local notation "L" => AddMonoid.End.mulLeft

local notation "R" => AddMonoid.End.mulRight


lemma centroid_eq_centralizer_mulLeftRight :
    RingHom.rangeS (toEndRingHom α) = Subsemiring.centralizer (Set.range L ∪ Set.range R) := by
  /-
    α : Type u_5
    inst✝ : NonUnitalNonAssocSemiring α
    ⊢ Eq (CentroidHom.toEndRingHom α).rangeS (Subsemiring.centralizer (Union.union …
  -/
  ext T
  /-
    case h
    α : Type u_5
    inst✝ : NonUnitalNonAssocSemiring α
    T : AddMonoid.End α
    ⊢ Iff (Membership.mem (CentroidHom.toEndRingHom α).rangeS T) (Membership.mem ( …
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case h.refine_1
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      T : AddMonoid.End α
      ⊢ Membership.mem (CentroidHom.toEndRingHom α).rangeS T → Membership.mem (Subse …
    -/
  · rintro ⟨f, rfl⟩ S (⟨a, rfl⟩ | ⟨b, rfl⟩)
      /-
        case h.refine_1.intro.inl.intro
        α : Type u_5
        inst✝ : NonUnitalNonAssocSemiring α
        f : CentroidHom α
        a : α
        ⊢ Eq (HMul.hMul (AddMonoid.End.mulLeft a) ((CentroidHom.toEndRingHom α) f)) (H …
      -/
    · exact AddMonoidHom.ext fun b ↦ (map_mul_left f a b).symm
      /-
        🎉 no goals
      -/
      /-
        case h.refine_1.intro.inr.intro
        α : Type u_5
        inst✝ : NonUnitalNonAssocSemiring α
        f : CentroidHom α
        b : α
        ⊢ Eq (HMul.hMul (AddMonoid.End.mulRight b) ((CentroidHom.toEndRingHom α) f)) ( …
      -/
    · exact AddMonoidHom.ext fun a ↦ (map_mul_right f a b).symm
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      T : AddMonoid.End α
      h : Membership.mem (Subsemiring.centralizer (Union.union (Set.range ⇑AddMonoid …
      ⊢ Membership.mem (CentroidHom.toEndRingHom α).rangeS T
    -/
  · rw [Subsemiring.mem_centralizer_iff] at h
    /-
      case h.refine_2
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      T : AddMonoid.End α
      h : ∀ (g : AddMonoid.End α), Membership.mem (Union.union (Set.range ⇑AddMonoid …
      ⊢ Membership.mem (CentroidHom.toEndRingHom α).rangeS T
    -/
    refine ⟨⟨T, fun a b ↦ ?_, fun a b ↦ ?_⟩, rfl⟩
      /-
        case h.refine_2.refine_1
        α : Type u_5
        inst✝ : NonUnitalNonAssocSemiring α
        T : AddMonoid.End α
        h : ∀ (g : AddMonoid.End α), Membership.mem (Union.union (Set.range ⇑AddMonoid …
        a b : α
        ⊢ Eq ((↑T).toFun (HMul.hMul a b)) (HMul.hMul a ((↑T).toFun b))
      -/
    · exact congr($(h (L a) (.inl ⟨a, rfl⟩)) b).symm
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.refine_2
        α : Type u_5
        inst✝ : NonUnitalNonAssocSemiring α
        T : AddMonoid.End α
        h : ∀ (g : AddMonoid.End α), Membership.mem (Union.union (Set.range ⇑AddMonoid …
        a b : α
        ⊢ Eq ((↑T).toFun (HMul.hMul a b)) (HMul.hMul ((↑T).toFun a) b)
      -/
    · exact congr($(h (R b) (.inr ⟨b, rfl⟩)) a).symm
      /-
        🎉 no goals
      -/


/-- The canonical homomorphism from the center into the center of the centroid -/
def centerToCentroidCenter :
    NonUnitalSubsemiring.center α →ₙ+* Subsemiring.center (CentroidHom α) where
  toFun z :=
    { L (z : α) with
      val := ⟨L z, z.prop.left_comm, z.prop.left_assoc ⟩
      property := by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R✝ : Type u_4
          α : Type u_5
          inst✝¹⁶ : NonUnitalNonAssocSemiring α
          inst✝¹⁵ : Monoid M
          inst✝¹⁴ : Monoid N
          inst✝¹³ : Semiring R✝
          inst✝¹² : DistribMulAction M α
          inst✝¹¹ : SMulCommClass M α α
          inst✝¹⁰ : IsScalarTower M α α
          inst✝⁹ : DistribMulAction N α
          inst✝⁸ : SMulCommClass N α α
          inst✝⁷ : IsScalarTower N α α
          inst✝⁶ : Module R✝ α
          inst✝⁵ : SMulCommClass R✝ α α
          inst✝⁴ : IsScalarTower R✝ α α
          R : Type u_6
          inst✝³ : CommSemiring «R»
          inst✝² : Module «R» α
          inst✝¹ : SMulCommClass «R» α α
          inst✝ : IsScalarTower «R» α α
          z : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
          ⊢ Membership.mem (Subsemiring.center (CentroidHom α)) { toAddMonoidHom := AddM …
        -/
        rw [Subsemiring.mem_center_iff]
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R✝ : Type u_4
          α : Type u_5
          inst✝¹⁶ : NonUnitalNonAssocSemiring α
          inst✝¹⁵ : Monoid M
          inst✝¹⁴ : Monoid N
          inst✝¹³ : Semiring R✝
          inst✝¹² : DistribMulAction M α
          inst✝¹¹ : SMulCommClass M α α
          inst✝¹⁰ : IsScalarTower M α α
          inst✝⁹ : DistribMulAction N α
          inst✝⁸ : SMulCommClass N α α
          inst✝⁷ : IsScalarTower N α α
          inst✝⁶ : Module R✝ α
          inst✝⁵ : SMulCommClass R✝ α α
          inst✝⁴ : IsScalarTower R✝ α α
          R : Type u_6
          inst✝³ : CommSemiring «R»
          inst✝² : Module «R» α
          inst✝¹ : SMulCommClass «R» α α
          inst✝ : IsScalarTower «R» α α
          z : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
          ⊢ ∀ (g : CentroidHom α), Eq (HMul.hMul g { toAddMonoidHom := AddMonoid.End.mul …
        -/
        intros g
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R✝ : Type u_4
          α : Type u_5
          inst✝¹⁶ : NonUnitalNonAssocSemiring α
          inst✝¹⁵ : Monoid M
          inst✝¹⁴ : Monoid N
          inst✝¹³ : Semiring R✝
          inst✝¹² : DistribMulAction M α
          inst✝¹¹ : SMulCommClass M α α
          inst✝¹⁰ : IsScalarTower M α α
          inst✝⁹ : DistribMulAction N α
          inst✝⁸ : SMulCommClass N α α
          inst✝⁷ : IsScalarTower N α α
          inst✝⁶ : Module R✝ α
          inst✝⁵ : SMulCommClass R✝ α α
          inst✝⁴ : IsScalarTower R✝ α α
          R : Type u_6
          inst✝³ : CommSemiring «R»
          inst✝² : Module «R» α
          inst✝¹ : SMulCommClass «R» α α
          inst✝ : IsScalarTower «R» α α
          z : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
          g : CentroidHom α
          ⊢ Eq (HMul.hMul g { toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left'  …
        -/
        ext a
        /-
          case h
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R✝ : Type u_4
          α : Type u_5
          inst✝¹⁶ : NonUnitalNonAssocSemiring α
          inst✝¹⁵ : Monoid M
          inst✝¹⁴ : Monoid N
          inst✝¹³ : Semiring R✝
          inst✝¹² : DistribMulAction M α
          inst✝¹¹ : SMulCommClass M α α
          inst✝¹⁰ : IsScalarTower M α α
          inst✝⁹ : DistribMulAction N α
          inst✝⁸ : SMulCommClass N α α
          inst✝⁷ : IsScalarTower N α α
          inst✝⁶ : Module R✝ α
          inst✝⁵ : SMulCommClass R✝ α α
          inst✝⁴ : IsScalarTower R✝ α α
          R : Type u_6
          inst✝³ : CommSemiring «R»
          inst✝² : Module «R» α
          inst✝¹ : SMulCommClass «R» α α
          inst✝ : IsScalarTower «R» α α
          z : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
          g : CentroidHom α
          a : α
          ⊢ Eq ((HMul.hMul g { toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' …
        -/
        exact map_mul_left g (↑z) a }
        /-
          🎉 no goals
        -/
  map_zero' := by
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      ⊢ Eq
          ({
                toFun := fun z =>
                  let __src := AddMonoid.End.mulLeft ↑z;
                  ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' := ⋯, …
                map_mul' := ⋯ }.toFun
            0)
          0
    -/
    simp only [ZeroMemClass.coe_zero, map_zero]
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      ⊢ Eq ⟨{ toAddMonoidHom := 0, map_mul_left' := ⋯, map_mul_right' := ⋯ }, ⋯⟩ 0
    -/
    exact rfl
    /-
      🎉 no goals
    -/
  map_add' := fun _ _ => by
                       /-
                         F : Type u_1
                         M : Type u_2
                         N : Type u_3
                         R✝ : Type u_4
                         α : Type u_5
                         inst✝¹⁶ : NonUnitalNonAssocSemiring α
                         inst✝¹⁵ : Monoid M
                         inst✝¹⁴ : Monoid N
                         inst✝¹³ : Semiring R✝
                         inst✝¹² : DistribMulAction M α
                         inst✝¹¹ : SMulCommClass M α α
                         inst✝¹⁰ : IsScalarTower M α α
                         inst✝⁹ : DistribMulAction N α
                         inst✝⁸ : SMulCommClass N α α
                         inst✝⁷ : IsScalarTower N α α
                         inst✝⁶ : Module R✝ α
                         inst✝⁵ : SMulCommClass R✝ α α
                         inst✝⁴ : IsScalarTower R✝ α α
                         R : Type u_6
                         inst✝³ : CommSemiring «R»
                         inst✝² : Module «R» α
                         inst✝¹ : SMulCommClass «R» α α
                         inst✝ : IsScalarTower «R» α α
                         z₁ z₂ : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
                         ⊢ Eq
                             ((fun z =>
                                 let __src := AddMonoid.End.mulLeft ↑z;
                                 ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' := ⋯, map …
                               (HMul.hMul z₁ z₂))
                             (HMul.hMul
                               ((fun z =>
                                   let __src := AddMonoid.End.mulLeft ↑z;
                                   ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' := ⋯, m …
                                 z₁)
                               ((fun z =>
                                   let __src := AddMonoid.End.mulLeft ↑z;
                                   ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' := ⋯, m …
                                 z₂))
                       -/
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
      ⊢ Eq
          ({
                toFun := fun z =>
                  let __src := AddMonoid.End.mulLeft ↑z;
                  ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' := ⋯, …
                map_mul' := ⋯ }.toFun
            (HAdd.hAdd x✝¹ x✝))
          (HAdd.hAdd
            ({
                  toFun := fun z =>
                    let __src := AddMonoid.End.mulLeft ↑z;
                    ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' :=  …
                  map_mul' := ⋯ }.toFun
              x✝¹)
            ({
                  toFun := fun z =>
                    let __src := AddMonoid.End.mulLeft ↑z;
                    ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft ↑z, map_mul_left' :=  …
                  map_mul' := ⋯ }.toFun
              x✝))
    -/
                              /-
                                🎉 no goals
                              -/
    dsimp
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
      ⊢ Eq ⟨{ toAddMonoidHom := AddMonoid.End.mulLeft (HAdd.hAdd ↑x✝¹ ↑x✝), map_mul_ …
    -/
    simp only [map_add]
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      x✝¹ x✝ : Subtype fun x => Membership.mem (NonUnitalSubsemiring.center α) x
      ⊢ Eq ⟨{ toAddMonoidHom := HAdd.hAdd (AddMonoid.End.mulLeft ↑x✝¹) (AddMonoid.En …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_mul' z₁ z₂ := by ext a; exact (z₁.prop.left_assoc z₂ a).symm


instance : FunLike (Subsemiring.center (CentroidHom α)) α α where
  coe f := f.val.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      f g : Subtype fun x => Membership.mem (Subsemiring.center (CentroidHom α)) x
      h : Eq ((fun f => (↑(↑f).toAddMonoidHom).toFun) f) ((fun f => (↑(↑f).toAddMono …
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      g : Subtype fun x => Membership.mem (Subsemiring.center (CentroidHom α)) x
      val✝ : CentroidHom α
      property✝ : Membership.mem (Subsemiring.center (CentroidHom α)) val✝
      h : Eq ((fun f => (↑(↑f).toAddMonoidHom).toFun) ⟨val✝, property✝⟩) ((fun f =>  …
      ⊢ Eq ⟨val✝, property✝⟩ g
    -/
    cases g
    /-
      case mk.mk
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      val✝¹ : CentroidHom α
      property✝¹ : Membership.mem (Subsemiring.center (CentroidHom α)) val✝¹
      val✝ : CentroidHom α
      property✝ : Membership.mem (Subsemiring.center (CentroidHom α)) val✝
      h : Eq ((fun f => (↑(↑f).toAddMonoidHom).toFun) ⟨val✝¹, property✝¹⟩) ((fun f = …
      ⊢ Eq ⟨val✝¹, property✝¹⟩ ⟨val✝, property✝⟩
    -/
    congr with x
    /-
      case mk.mk.e_val.h
      F : Type u_1
      M : Type u_2
      N : Type u_3
      R✝ : Type u_4
      α : Type u_5
      inst✝¹⁶ : NonUnitalNonAssocSemiring α
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : Monoid N
      inst✝¹³ : Semiring R✝
      inst✝¹² : DistribMulAction M α
      inst✝¹¹ : SMulCommClass M α α
      inst✝¹⁰ : IsScalarTower M α α
      inst✝⁹ : DistribMulAction N α
      inst✝⁸ : SMulCommClass N α α
      inst✝⁷ : IsScalarTower N α α
      inst✝⁶ : Module R✝ α
      inst✝⁵ : SMulCommClass R✝ α α
      inst✝⁴ : IsScalarTower R✝ α α
      R : Type u_6
      inst✝³ : CommSemiring «R»
      inst✝² : Module «R» α
      inst✝¹ : SMulCommClass «R» α α
      inst✝ : IsScalarTower «R» α α
      val✝¹ : CentroidHom α
      property✝¹ : Membership.mem (Subsemiring.center (CentroidHom α)) val✝¹
      val✝ : CentroidHom α
      property✝ : Membership.mem (Subsemiring.center (CentroidHom α)) val✝
      h : Eq ((fun f => (↑(↑f).toAddMonoidHom).toFun) ⟨val✝¹, property✝¹⟩) ((fun f = …
      x : α
      ⊢ Eq (val✝¹ x) (val✝ x)
    -/
    exact congrFun h x
    /-
      🎉 no goals
    -/


lemma centerToCentroidCenter_apply (z : NonUnitalSubsemiring.center α) (a : α) :
    (centerToCentroidCenter z) a = z * a := rfl


/-- The canonical homomorphism from the center into the centroid -/
def centerToCentroid : NonUnitalSubsemiring.center α →ₙ+* CentroidHom α :=
  NonUnitalRingHom.comp
    (SubsemiringClass.subtype (Subsemiring.center (CentroidHom α))).toNonUnitalRingHom
    centerToCentroidCenter


lemma centerToCentroid_apply (z : NonUnitalSubsemiring.center α) (a : α) :
    (centerToCentroid z) a = z * a := rfl


lemma _root_.NonUnitalNonAssocSemiring.mem_center_iff (a : α) :
    a ∈ NonUnitalSubsemiring.center α ↔ R a = L a ∧ (L a) ∈ RingHom.rangeS (toEndRingHom α) := by
  /-
    α : Type u_5
    inst✝ : NonUnitalNonAssocSemiring α
    a : α
    ⊢ Iff (Membership.mem (NonUnitalSubsemiring.center α) a) (And (Eq (AddMonoid.E …
  -/
  constructor
  · exact fun ha ↦ ⟨AddMonoidHom.ext <| fun _ => (IsMulCentral.comm ha _).symm,
      ⟨centerToCentroid ⟨a, ha⟩, rfl⟩⟩
    /-
      case mpr
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      ⊢ And (Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)) (Membership.me …
    -/
  · rintro ⟨hc, ⟨T, hT⟩⟩
    /-
      case mpr.intro.intro
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      hc : Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)
      T : CentroidHom α
      hT : Eq ((CentroidHom.toEndRingHom α) T) (AddMonoid.End.mulLeft a)
      ⊢ Membership.mem (NonUnitalSubsemiring.center α) a
    -/
    have e1 (d : α) : T d = a * d := congr($hT d)
    /-
      case mpr.intro.intro
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      hc : Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)
      T : CentroidHom α
      hT : Eq ((CentroidHom.toEndRingHom α) T) (AddMonoid.End.mulLeft a)
      e1 : ∀ (d : α), Eq (T d) (HMul.hMul a d)
      ⊢ Membership.mem (NonUnitalSubsemiring.center α) a
    -/
    have e2 (d : α) : T d = d * a := congr($(hT.trans hc.symm) d)
    /-
      case mpr.intro.intro
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      hc : Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)
      T : CentroidHom α
      hT : Eq ((CentroidHom.toEndRingHom α) T) (AddMonoid.End.mulLeft a)
      e1 : ∀ (d : α), Eq (T d) (HMul.hMul a d)
      e2 : ∀ (d : α), Eq (T d) (HMul.hMul d a)
      ⊢ Membership.mem (NonUnitalSubsemiring.center α) a
    -/
    constructor
    /-
      case mpr.intro.intro.comm
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      hc : Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)
      T : CentroidHom α
      hT : Eq ((CentroidHom.toEndRingHom α) T) (AddMonoid.End.mulLeft a)
      e1 : ∀ (d : α), Eq (T d) (HMul.hMul a d)
      e2 : ∀ (d : α), Eq (T d) (HMul.hMul d a)
      ⊢ ∀ (a_1 : α), Eq (HMul.hMul a a_1) (HMul.hMul a_1 a)
    -/
    case comm => exact (congr($hc.symm ·))
    /-
      case mpr.intro.intro.left_assoc
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      hc : Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)
      T : CentroidHom α
      hT : Eq ((CentroidHom.toEndRingHom α) T) (AddMonoid.End.mulLeft a)
      e1 : ∀ (d : α), Eq (T d) (HMul.hMul a d)
      e2 : ∀ (d : α), Eq (T d) (HMul.hMul d a)
      ⊢ ∀ (b c : α), Eq (HMul.hMul a (HMul.hMul b c)) (HMul.hMul (HMul.hMul a b) c)
    -/
    case left_assoc => simpa [e1] using (map_mul_right T · ·)
    case mid_assoc => exact fun b c ↦ by simpa [e1 c, e2 b] using
      (map_mul_right T b c).symm.trans <| map_mul_left T b c
    /-
      case mpr.intro.intro.right_assoc
      α : Type u_5
      inst✝ : NonUnitalNonAssocSemiring α
      a : α
      hc : Eq (AddMonoid.End.mulRight a) (AddMonoid.End.mulLeft a)
      T : CentroidHom α
      hT : Eq ((CentroidHom.toEndRingHom α) T) (AddMonoid.End.mulLeft a)
      e1 : ∀ (d : α), Eq (T d) (HMul.hMul a d)
      e2 : ∀ (d : α), Eq (T d) (HMul.hMul d a)
      ⊢ ∀ (a_1 b : α), Eq (HMul.hMul (HMul.hMul a_1 b) a) (HMul.hMul a_1 (HMul.hMul  …
    -/
    case right_assoc => simpa [e2] using (map_mul_left T · ·)
    /-
      🎉 no goals
    -/


local notation "L" => AddMonoid.End.mulLeft


lemma _root_.NonUnitalNonAssocCommSemiring.mem_center_iff (a : α) :
    a ∈ NonUnitalSubsemiring.center α ↔ ∀ b : α, Commute (L b) (L a) := by
  rw [NonUnitalNonAssocSemiring.mem_center_iff, CentroidHom.centroid_eq_centralizer_mulLeftRight,
    Subsemiring.mem_centralizer_iff, AddMonoid.End.mulRight_eq_mulLeft, Set.union_self]
  /-
    α : Type u_5
    inst✝ : NonUnitalNonAssocCommSemiring α
    a : α
    ⊢ Iff (And (Eq (AddMonoid.End.mulLeft a) (AddMonoid.End.mulLeft a)) (∀ (g : Ad …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism from the center of a (non-associative) semiring onto its centroid. -/
def centerIsoCentroid : Subsemiring.center α ≃+* CentroidHom α :=
  { centerToCentroid with
    invFun := fun T ↦
               /-
                 F : Type u_1
                 M : Type u_2
                 N : Type u_3
                 R : Type u_4
                 α : Type u_5
                 inst✝ : NonAssocSemiring α
                 T : CentroidHom α
                 ⊢ Membership.mem (Subsemiring.center α) (T 1)
               -/
      ⟨T 1, by refine ⟨?_, ?_, ?_, ?_⟩; all_goals simp [← map_mul_left, ← map_mul_right]⟩
                                        /-
                                          🎉 no goals
                                        -/
    left_inv := fun z ↦ Subtype.ext <| by simp only [MulHom.toFun_eq_coe,
      NonUnitalRingHom.coe_toMulHom, centerToCentroid_apply, mul_one]
    right_inv := fun T ↦ CentroidHom.ext <| fun _ => by rw [MulHom.toFun_eq_coe,
      NonUnitalRingHom.coe_toMulHom, centerToCentroid_apply, ← map_mul_right, one_mul] }


/-- Negation of `CentroidHom`s as a `CentroidHom`. -/
instance : Neg (CentroidHom α) :=
  ⟨fun f ↦
    { (-f : α →+ α) with
      map_mul_left' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul a ((↑__src✝).toFun b))
        -/
        change -f (a * b) = a * (-f b)
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f : CentroidHom α
          a b : α
          ⊢ Eq (Neg.neg (f (HMul.hMul a b))) (HMul.hMul a (Neg.neg (f b)))
        -/
        simp [map_mul_left]
        /-
          🎉 no goals
        -/
      map_mul_right' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul ((↑__src✝).toFun a) b)
        -/
        change -f (a * b) = (-f a) * b
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f : CentroidHom α
          a b : α
          ⊢ Eq (Neg.neg (f (HMul.hMul a b))) (HMul.hMul (Neg.neg (f a)) b)
        -/
        simp [map_mul_right] }⟩
        /-
          🎉 no goals
        -/


instance : Sub (CentroidHom α) :=
  ⟨fun f g ↦
    { (f - g : α →+ α) with
      map_mul_left' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f g : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul a ((↑__src✝).toFun b))
        -/
        change (⇑f - ⇑g) (a * b) = a * (⇑f - ⇑g) b
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f g : CentroidHom α
          a b : α
          ⊢ Eq (HSub.hSub (⇑f) (⇑g) (HMul.hMul a b)) (HMul.hMul a (HSub.hSub (⇑f) (⇑g) b))
        -/
        simp [map_mul_left, mul_sub]
        /-
          🎉 no goals
        -/
      map_mul_right' := fun a b ↦ by
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f g : CentroidHom α
          a b : α
          ⊢ Eq ((↑__src✝).toFun (HMul.hMul a b)) (HMul.hMul ((↑__src✝).toFun a) b)
        -/
        change (⇑f - ⇑g) (a * b) = ((⇑f - ⇑g) a) * b
        /-
          F : Type u_1
          M : Type u_2
          N : Type u_3
          R : Type u_4
          α : Type u_5
          inst✝ : NonUnitalNonAssocRing α
          f g : CentroidHom α
          a b : α
          ⊢ Eq (HSub.hSub (⇑f) (⇑g) (HMul.hMul a b)) (HMul.hMul (HSub.hSub (⇑f) (⇑g) a) b)
        -/
        simp [map_mul_right, sub_mul] }⟩
        /-
          🎉 no goals
        -/


instance : IntCast (CentroidHom α) where intCast z := z • (1 : CentroidHom α)


@[simp, norm_cast]
theorem coe_intCast (z : ℤ) : ⇑(z : CentroidHom α) = z • (CentroidHom.id α) :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_int_cast := coe_intCast


theorem intCast_apply (z : ℤ) (m : α) : (z : CentroidHom α) m = z • m :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_apply := intCast_apply


@[simp]
theorem toEnd_neg (x : CentroidHom α) : (-x).toEnd = -x.toEnd :=
  rfl


@[simp]
theorem toEnd_sub (x y : CentroidHom α) : (x - y).toEnd = x.toEnd - y.toEnd :=
  rfl


instance : AddCommGroup (CentroidHom α) :=
  toEnd_injective.addCommGroup _
    toEnd_zero toEnd_add toEnd_neg toEnd_sub (swap toEnd_smul) (swap toEnd_smul)


@[simp, norm_cast]
theorem coe_neg (f : CentroidHom α) : ⇑(-f) = -f :=
  rfl


@[simp, norm_cast]
theorem coe_sub (f g : CentroidHom α) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem neg_apply (f : CentroidHom α) (a : α) : (-f) a = -f a :=
  rfl


@[simp]
theorem sub_apply (f g : CentroidHom α) (a : α) : (f - g) a = f a - g a :=
  rfl


@[simp, norm_cast]
theorem toEnd_intCast (z : ℤ) : (z : CentroidHom α).toEnd = ↑z :=
  rfl


@[deprecated (since := "2024-04-17")]
alias toEnd_int_cast := toEnd_intCast


instance instRing : Ring (CentroidHom α) :=
  toEnd_injective.ring _ toEnd_zero toEnd_one toEnd_add toEnd_mul toEnd_neg toEnd_sub
    toEnd_smul toEnd_smul toEnd_pow toEnd_natCast toEnd_intCast


/-- A prime associative ring has commutative centroid. -/
abbrev commRing
    (h : ∀ a b : α, (∀ r : α, a * r * b = 0) → a = 0 ∨ b = 0) : CommRing (CentroidHom α) :=
  { CentroidHom.instRing with
    mul_comm := fun f g ↦ by
      /-
        F : Type u_1
        M : Type u_2
        N : Type u_3
        R : Type u_4
        α : Type u_5
        inst✝ : NonUnitalRing α
        h : ∀ (a b : α), (∀ (r : α), Eq (HMul.hMul (HMul.hMul a r) b) 0) → Or (Eq a 0) …
        f g : CentroidHom α
        ⊢ Eq (HMul.hMul f g) (HMul.hMul g f)
      -/
      ext
      /-
        case h
        F : Type u_1
        M : Type u_2
        N : Type u_3
        R : Type u_4
        α : Type u_5
        inst✝ : NonUnitalRing α
        h : ∀ (a b : α), (∀ (r : α), Eq (HMul.hMul (HMul.hMul a r) b) 0) → Or (Eq a 0) …
        f g : CentroidHom α
        a✝ : α
        ⊢ Eq ((HMul.hMul f g) a✝) ((HMul.hMul g f) a✝)
      -/
      refine sub_eq_zero.1 (or_self_iff.1 <| (h _ _) fun r ↦ ?_)
      rw [mul_assoc, sub_mul, sub_eq_zero, ← map_mul_right, ← map_mul_right, coe_mul, coe_mul,
        comp_mul_comm] }


