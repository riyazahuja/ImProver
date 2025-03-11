/-- An element is self-adjoint if it is equal to its star. -/
def IsSelfAdjoint [Star R] (x : R) : Prop :=
  star x = x


/-- An element of a star monoid is normal if it commutes with its adjoint. -/
@[mk_iff]
class IsStarNormal [Mul R] [Star R] (x : R) : Prop where
  /-- A normal element of a star monoid commutes with its adjoint. -/
  star_comm_self : Commute (star x) x


theorem star_comm_self' [Mul R] [Star R] (x : R) [IsStarNormal x] : star x * x = x * star x :=
  IsStarNormal.star_comm_self


/-- All elements are self-adjoint when `star` is trivial. -/
theorem all [Star R] [TrivialStar R] (r : R) : IsSelfAdjoint r :=
  star_trivial _


theorem star_eq [Star R] {x : R} (hx : IsSelfAdjoint x) : star x = x :=
  hx


theorem _root_.isSelfAdjoint_iff [Star R] {x : R} : IsSelfAdjoint x ↔ star x = x :=
  Iff.rfl


@[simp]
theorem star_iff [InvolutiveStar R] {x : R} : IsSelfAdjoint (star x) ↔ IsSelfAdjoint x := by
  /-
    R : Type u_1
    inst✝ : InvolutiveStar R
    x : R
    ⊢ Iff (IsSelfAdjoint (Star.star x)) (IsSelfAdjoint x)
  -/
  simpa only [IsSelfAdjoint, star_star] using eq_comm
  /-
    🎉 no goals
  -/


@[simp]
theorem star_mul_self [Mul R] [StarMul R] (x : R) : IsSelfAdjoint (star x * x) := by
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    x : R
    ⊢ IsSelfAdjoint (HMul.hMul (Star.star x) x)
  -/
  simp only [IsSelfAdjoint, star_mul, star_star]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_star_self [Mul R] [StarMul R] (x : R) : IsSelfAdjoint (x * star x) := by
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    x : R
    ⊢ IsSelfAdjoint (HMul.hMul x (Star.star x))
  -/
  simpa only [star_star] using star_mul_self (star x)
  /-
    🎉 no goals
  -/


/-- Self-adjoint elements commute if and only if their product is self-adjoint. -/
lemma commute_iff {R : Type*} [Mul R] [StarMul R] {x y : R}
    (hx : IsSelfAdjoint x) (hy : IsSelfAdjoint y) : Commute x y ↔ IsSelfAdjoint (x * y) := by
  /-
    R : Type u_3
    inst✝¹ : Mul R
    inst✝ : StarMul R
    x y : R
    hx : IsSelfAdjoint x
    hy : IsSelfAdjoint y
    ⊢ Iff (Commute x y) (IsSelfAdjoint (HMul.hMul x y))
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_3
      inst✝¹ : Mul R
      inst✝ : StarMul R
      x y : R
      hx : IsSelfAdjoint x
      hy : IsSelfAdjoint y
      h : Commute x y
      ⊢ IsSelfAdjoint (HMul.hMul x y)
    -/
  · rw [isSelfAdjoint_iff, star_mul, hx.star_eq, hy.star_eq, h.eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_3
      inst✝¹ : Mul R
      inst✝ : StarMul R
      x y : R
      hx : IsSelfAdjoint x
      hy : IsSelfAdjoint y
      h : IsSelfAdjoint (HMul.hMul x y)
      ⊢ Commute x y
    -/
  · simpa only [star_mul, hx.star_eq, hy.star_eq] using h.symm
    /-
      🎉 no goals
    -/


/-- Functions in a `StarHomClass` preserve self-adjoint elements. -/
@[aesop 10% apply]
theorem map {F R S : Type*} [Star R] [Star S] [FunLike F R S] [StarHomClass F R S]
    {x : R} (hx : IsSelfAdjoint x) (f : F) : IsSelfAdjoint (f x) :=
  show star (f x) = f x from map_star f x ▸ congr_arg f hx


@[deprecated (since := "2024-09-07")] alias starHom_apply := map

/- note: this lemma is *not* marked as `simp` so that Lean doesn't look for a `[TrivialStar R]`
instance every time it sees `⊢ IsSelfAdjoint (f x)`, which will likely occur relatively often. -/

theorem _root_.isSelfAdjoint_map {F R S : Type*} [Star R] [Star S] [FunLike F R S]
    [StarHomClass F R S] [TrivialStar R] (f : F) (x : R) : IsSelfAdjoint (f x) :=
  (IsSelfAdjoint.all x).map f


@[deprecated (since := "2024-09-07")] alias _root_.isSelfAdjoint_starHom_apply := isSelfAdjoint_map


@[simp] protected theorem zero : IsSelfAdjoint (0 : R) := star_zero R


@[aesop 90% apply]
theorem add {x y : R} (hx : IsSelfAdjoint x) (hy : IsSelfAdjoint y) : IsSelfAdjoint (x + y) := by
  /-
    R : Type u_1
    inst✝¹ : AddMonoid R
    inst✝ : StarAddMonoid R
    x y : R
    hx : IsSelfAdjoint x
    hy : IsSelfAdjoint y
    ⊢ IsSelfAdjoint (HAdd.hAdd x y)
  -/
  simp only [isSelfAdjoint_iff, star_add, hx.star_eq, hy.star_eq]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem neg {x : R} (hx : IsSelfAdjoint x) : IsSelfAdjoint (-x) := by
  /-
    R : Type u_1
    inst✝¹ : AddGroup R
    inst✝ : StarAddMonoid R
    x : R
    hx : IsSelfAdjoint x
    ⊢ IsSelfAdjoint (Neg.neg x)
  -/
  simp only [isSelfAdjoint_iff, star_neg, hx.star_eq]
  /-
    🎉 no goals
  -/


@[aesop 90% apply]
theorem sub {x y : R} (hx : IsSelfAdjoint x) (hy : IsSelfAdjoint y) : IsSelfAdjoint (x - y) := by
  /-
    R : Type u_1
    inst✝¹ : AddGroup R
    inst✝ : StarAddMonoid R
    x y : R
    hx : IsSelfAdjoint x
    hy : IsSelfAdjoint y
    ⊢ IsSelfAdjoint (HSub.hSub x y)
  -/
  simp only [isSelfAdjoint_iff, star_sub, hx.star_eq, hy.star_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_star_self (x : R) : IsSelfAdjoint (x + star x) := by
  /-
    R : Type u_1
    inst✝¹ : AddCommMonoid R
    inst✝ : StarAddMonoid R
    x : R
    ⊢ IsSelfAdjoint (HAdd.hAdd x (Star.star x))
  -/
  simp only [isSelfAdjoint_iff, add_comm, star_add, star_star]
  /-
    🎉 no goals
  -/


@[simp]
theorem star_add_self (x : R) : IsSelfAdjoint (star x + x) := by
  /-
    R : Type u_1
    inst✝¹ : AddCommMonoid R
    inst✝ : StarAddMonoid R
    x : R
    ⊢ IsSelfAdjoint (HAdd.hAdd (Star.star x) x)
  -/
  simp only [isSelfAdjoint_iff, add_comm, star_add, star_star]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem conjugate {x : R} (hx : IsSelfAdjoint x) (z : R) : IsSelfAdjoint (z * x * star z) := by
  /-
    R : Type u_1
    inst✝¹ : Semigroup R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    z : R
    ⊢ IsSelfAdjoint (HMul.hMul (HMul.hMul z x) (Star.star z))
  -/
  simp only [isSelfAdjoint_iff, star_mul, star_star, mul_assoc, hx.star_eq]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem conjugate' {x : R} (hx : IsSelfAdjoint x) (z : R) : IsSelfAdjoint (star z * x * z) := by
  /-
    R : Type u_1
    inst✝¹ : Semigroup R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    z : R
    ⊢ IsSelfAdjoint (HMul.hMul (HMul.hMul (Star.star z) x) z)
  -/
  simp only [isSelfAdjoint_iff, star_mul, star_star, mul_assoc, hx.star_eq]
  /-
    🎉 no goals
  -/


@[aesop 90% apply]
theorem conjugate_self {x : R} (hx : IsSelfAdjoint x) {z : R} (hz : IsSelfAdjoint z) :
                                    /-
                                      R : Type u_1
                                      inst✝¹ : Semigroup R
                                      inst✝ : StarMul R
                                      x : R
                                      hx : IsSelfAdjoint x
                                      z : R
                                      hz : IsSelfAdjoint z
                                      ⊢ IsSelfAdjoint (HMul.hMul (HMul.hMul z x) z)
                                    -/
    IsSelfAdjoint (z * x * z) := by nth_rewrite 2 [← hz]; exact conjugate hx z
                                                          /-
                                                            🎉 no goals
                                                          -/


@[aesop 10% apply]
theorem isStarNormal {x : R} (hx : IsSelfAdjoint x) : IsStarNormal x :=
      /-
        R : Type u_1
        inst✝¹ : Semigroup R
        inst✝ : StarMul R
        x : R
        hx : IsSelfAdjoint x
        ⊢ Commute (Star.star x) x
      -/
  ⟨by simp only [Commute, SemiconjBy, hx.star_eq]⟩
      /-
        🎉 no goals
      -/


@[simp] protected theorem one : IsSelfAdjoint (1 : R) :=
  star_one R


@[aesop safe apply]
theorem pow {x : R} (hx : IsSelfAdjoint x) (n : ℕ) : IsSelfAdjoint (x ^ n) := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    n : Nat
    ⊢ IsSelfAdjoint (HPow.hPow x n)
  -/
  simp only [isSelfAdjoint_iff, star_pow, hx.star_eq]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem natCast (n : ℕ) : IsSelfAdjoint (n : R) :=
  star_natCast _


@[simp]
protected theorem ofNat (n : ℕ) [n.AtLeastTwo] : IsSelfAdjoint (ofNat(n) : R) :=
  .natCast n


theorem mul {x y : R} (hx : IsSelfAdjoint x) (hy : IsSelfAdjoint y) : IsSelfAdjoint (x * y) := by
  /-
    R : Type u_1
    inst✝¹ : CommSemigroup R
    inst✝ : StarMul R
    x y : R
    hx : IsSelfAdjoint x
    hy : IsSelfAdjoint y
    ⊢ IsSelfAdjoint (HMul.hMul x y)
  -/
  simp only [isSelfAdjoint_iff, star_mul', hx.star_eq, hy.star_eq]
  /-
    🎉 no goals
  -/


lemma conj_eq (ha : IsSelfAdjoint a) : conj a = a := ha.star_eq


@[simp]
protected theorem intCast (z : ℤ) : IsSelfAdjoint (z : R) :=
  star_intCast _


@[aesop safe apply]
theorem inv {x : R} (hx : IsSelfAdjoint x) : IsSelfAdjoint x⁻¹ := by
  /-
    R : Type u_1
    inst✝¹ : Group R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    ⊢ IsSelfAdjoint (Inv.inv x)
  -/
  simp only [isSelfAdjoint_iff, star_inv, hx.star_eq]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem zpow {x : R} (hx : IsSelfAdjoint x) (n : ℤ) : IsSelfAdjoint (x ^ n) := by
  /-
    R : Type u_1
    inst✝¹ : Group R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    n : Int
    ⊢ IsSelfAdjoint (HPow.hPow x n)
  -/
  simp only [isSelfAdjoint_iff, star_zpow, hx.star_eq]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem inv₀ {x : R} (hx : IsSelfAdjoint x) : IsSelfAdjoint x⁻¹ := by
  /-
    R : Type u_1
    inst✝¹ : GroupWithZero R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    ⊢ IsSelfAdjoint (Inv.inv x)
  -/
  simp only [isSelfAdjoint_iff, star_inv₀, hx.star_eq]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem zpow₀ {x : R} (hx : IsSelfAdjoint x) (n : ℤ) : IsSelfAdjoint (x ^ n) := by
  /-
    R : Type u_1
    inst✝¹ : GroupWithZero R
    inst✝ : StarMul R
    x : R
    hx : IsSelfAdjoint x
    n : Int
    ⊢ IsSelfAdjoint (HPow.hPow x n)
  -/
  simp only [isSelfAdjoint_iff, star_zpow₀, hx.star_eq]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma nnratCast [DivisionSemiring R] [StarRing R] (q : ℚ≥0) :
    IsSelfAdjoint (q : R) :=
  star_nnratCast _


@[simp]
protected theorem ratCast (x : ℚ) : IsSelfAdjoint (x : R) :=
  star_ratCast _


theorem div {x y : R} (hx : IsSelfAdjoint x) (hy : IsSelfAdjoint y) : IsSelfAdjoint (x / y) := by
  /-
    R : Type u_1
    inst✝¹ : Semifield R
    inst✝ : StarRing R
    x y : R
    hx : IsSelfAdjoint x
    hy : IsSelfAdjoint y
    ⊢ IsSelfAdjoint (HDiv.hDiv x y)
  -/
  simp only [isSelfAdjoint_iff, star_div₀, hx.star_eq, hy.star_eq]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem smul [Star R] [AddMonoid A] [StarAddMonoid A] [SMul R A] [StarModule R A]
    {r : R} (hr : IsSelfAdjoint r) {x : A} (hx : IsSelfAdjoint x) :
    IsSelfAdjoint (r • x) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : Star R
    inst✝³ : AddMonoid A
    inst✝² : StarAddMonoid A
    inst✝¹ : SMul R A
    inst✝ : StarModule R A
    r : R
    hr : IsSelfAdjoint r
    x : A
    hx : IsSelfAdjoint x
    ⊢ IsSelfAdjoint (HSMul.hSMul r x)
  -/
  simp only [isSelfAdjoint_iff, star_smul, hr.star_eq, hx.star_eq]
  /-
    🎉 no goals
  -/


theorem smul_iff [Monoid R] [StarMul R] [AddMonoid A] [StarAddMonoid A]
    [MulAction R A] [StarModule R A] {r : R} (hr : IsSelfAdjoint r) (hu : IsUnit r) {x : A} :
    IsSelfAdjoint (r • x) ↔ IsSelfAdjoint x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : Monoid R
    inst✝⁴ : StarMul R
    inst✝³ : AddMonoid A
    inst✝² : StarAddMonoid A
    inst✝¹ : MulAction R A
    inst✝ : StarModule R A
    r : R
    hr : IsSelfAdjoint r
    hu : IsUnit r
    x : A
    ⊢ Iff (IsSelfAdjoint (HSMul.hSMul r x)) (IsSelfAdjoint x)
  -/
  refine ⟨fun hrx ↦ ?_, .smul hr⟩
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : Monoid R
    inst✝⁴ : StarMul R
    inst✝³ : AddMonoid A
    inst✝² : StarAddMonoid A
    inst✝¹ : MulAction R A
    inst✝ : StarModule R A
    r : R
    hr : IsSelfAdjoint r
    hu : IsUnit r
    x : A
    hrx : IsSelfAdjoint (HSMul.hSMul r x)
    ⊢ IsSelfAdjoint x
  -/
  lift r to Rˣ using hu
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝⁵ : Monoid R
    inst✝⁴ : StarMul R
    inst✝³ : AddMonoid A
    inst✝² : StarAddMonoid A
    inst✝¹ : MulAction R A
    inst✝ : StarModule R A
    x : A
    r : Units R
    hr : IsSelfAdjoint ↑r
    hrx : IsSelfAdjoint (HSMul.hSMul (↑r) x)
    ⊢ IsSelfAdjoint x
  -/
  rw [← inv_smul_smul r x]
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝⁵ : Monoid R
    inst✝⁴ : StarMul R
    inst✝³ : AddMonoid A
    inst✝² : StarAddMonoid A
    inst✝¹ : MulAction R A
    inst✝ : StarModule R A
    x : A
    r : Units R
    hr : IsSelfAdjoint ↑r
    hrx : IsSelfAdjoint (HSMul.hSMul (↑r) x)
    ⊢ IsSelfAdjoint (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r x))
  -/
  replace hr : IsSelfAdjoint r := Units.ext hr.star_eq
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝⁵ : Monoid R
    inst✝⁴ : StarMul R
    inst✝³ : AddMonoid A
    inst✝² : StarAddMonoid A
    inst✝¹ : MulAction R A
    inst✝ : StarModule R A
    x : A
    r : Units R
    hrx : IsSelfAdjoint (HSMul.hSMul (↑r) x)
    hr : IsSelfAdjoint r
    ⊢ IsSelfAdjoint (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r x))
  -/
  exact hr.inv.smul hrx
  /-
    🎉 no goals
  -/


/-- The self-adjoint elements of a star additive group, as an additive subgroup. -/
def selfAdjoint [AddGroup R] [StarAddMonoid R] : AddSubgroup R where
  carrier := { x | IsSelfAdjoint x }
  zero_mem' := star_zero R
  add_mem' hx := hx.add
  neg_mem' hx := hx.neg


/-- The skew-adjoint elements of a star additive group, as an additive subgroup. -/
def skewAdjoint [AddCommGroup R] [StarAddMonoid R] : AddSubgroup R where
  carrier := { x | star x = -x }
                                         /-
                                           R : Type u_1
                                           A : Type u_2
                                           inst✝¹ : AddCommGroup R
                                           inst✝ : StarAddMonoid R
                                           ⊢ Eq (Star.star 0) (-0)
                                         -/
  zero_mem' := show star (0 : R) = -0 by simp only [star_zero, neg_zero]
                                    /-
                                      R : Type u_1
                                      A : Type u_2
                                      inst✝¹ : AddCommGroup R
                                      inst✝ : StarAddMonoid R
                                      x y : R
                                      hx : Eq (Star.star x) (Neg.neg x)
                                      hy : Eq (Star.star y) (Neg.neg y)
                                      ⊢ Eq (Star.star (HAdd.hAdd x y)) (Neg.neg (HAdd.hAdd x y))
                                    -/
                                         /-
                                           🎉 no goals
                                         -/
                                    /-
                                      🎉 no goals
                                    -/
  add_mem' := @fun x y (hx : star x = -x) (hy : star y = -y) =>
    show star (x + y) = -(x + y) by rw [star_add x y, hx, hy, neg_add]
                                                                    /-
                                                                      R : Type u_1
                                                                      A : Type u_2
                                                                      inst✝¹ : AddCommGroup R
                                                                      inst✝ : StarAddMonoid R
                                                                      x : R
                                                                      hx : Eq (Star.star x) (Neg.neg x)
                                                                      ⊢ Eq (Star.star (Neg.neg x)) (Neg.neg (Neg.neg x))
                                                                    -/
  neg_mem' := @fun x (hx : star x = -x) => show star (-x) = - -x by simp only [hx, star_neg]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem mem_iff {x : R} : x ∈ selfAdjoint R ↔ star x = x := by
  /-
    R : Type u_1
    inst✝¹ : AddGroup R
    inst✝ : StarAddMonoid R
    x : R
    ⊢ Iff (Membership.mem (selfAdjoint R) x) (Eq (Star.star x) x)
  -/
  rw [← AddSubgroup.mem_carrier]
  /-
    R : Type u_1
    inst✝¹ : AddGroup R
    inst✝ : StarAddMonoid R
    x : R
    ⊢ Iff (Membership.mem (selfAdjoint R).carrier x) (Eq (Star.star x) x)
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem star_val_eq {x : selfAdjoint R} : star (x : R) = x :=
  x.prop


instance : Inhabited (selfAdjoint R) :=
  ⟨0⟩


instance isStarNormal [NonUnitalRing R] [StarRing R] (x : selfAdjoint R) :
    IsStarNormal (x : R) :=
  x.prop.isStarNormal


instance : One (selfAdjoint R) :=
  ⟨⟨1, .one R⟩⟩


@[simp, norm_cast]
theorem val_one : ↑(1 : selfAdjoint R) = (1 : R) :=
  rfl


instance [Nontrivial R] : Nontrivial (selfAdjoint R) :=
  ⟨⟨0, 1, ne_of_apply_ne Subtype.val zero_ne_one⟩⟩


instance : NatCast (selfAdjoint R) where
  natCast n := ⟨n, .natCast _⟩


instance : IntCast (selfAdjoint R) where
  intCast n := ⟨n, .intCast _⟩


instance : Pow (selfAdjoint R) ℕ where
  pow x n := ⟨(x : R) ^ n, x.prop.pow n⟩


@[simp, norm_cast]
theorem val_pow (x : selfAdjoint R) (n : ℕ) : ↑(x ^ n) = (x : R) ^ n :=
  rfl


instance : Mul (selfAdjoint R) where
  mul x y := ⟨(x : R) * y, x.prop.mul y.prop⟩


@[simp, norm_cast]
theorem val_mul (x y : selfAdjoint R) : ↑(x * y) = (x : R) * y :=
  rfl


instance : CommRing (selfAdjoint R) :=
  Function.Injective.commRing _ Subtype.coe_injective (selfAdjoint R).coe_zero val_one
    (selfAdjoint R).coe_add val_mul (selfAdjoint R).coe_neg (selfAdjoint R).coe_sub
        /-
          R : Type u_1
          A : Type u_2
          inst✝¹ : CommRing R
          inst✝ : StarRing R
          ⊢ ∀ (n : Nat) (x : Subtype fun x => Membership.mem (selfAdjoint R) x), Eq (↑(H …
        -/
                /-
                  🎉 no goals
                -/
    (by intros; rfl) (by intros; rfl) val_pow
                                 /-
                                   🎉 no goals
                                 -/
    (fun _ => rfl) fun _ => rfl


instance : Inv (selfAdjoint R) where
  inv x := ⟨x.val⁻¹, x.prop.inv₀⟩


@[simp, norm_cast]
theorem val_inv (x : selfAdjoint R) : ↑x⁻¹ = (x : R)⁻¹ :=
  rfl


instance : Div (selfAdjoint R) where
  div x y := ⟨x / y, x.prop.div y.prop⟩


@[simp, norm_cast]
theorem val_div (x y : selfAdjoint R) : ↑(x / y) = (x / y : R) :=
  rfl


instance : Pow (selfAdjoint R) ℤ where
  pow x z := ⟨(x : R) ^ z, x.prop.zpow₀ z⟩


@[simp, norm_cast]
theorem val_zpow (x : selfAdjoint R) (z : ℤ) : ↑(x ^ z) = (x : R) ^ z :=
  rfl


instance instNNRatCast : NNRatCast (selfAdjoint R) where
  nnratCast q := ⟨q, .nnratCast q⟩


instance instRatCast : RatCast (selfAdjoint R) where
  ratCast q := ⟨q, .ratCast q⟩


@[simp, norm_cast] lemma val_nnratCast (q : ℚ≥0) : (q : selfAdjoint R) = (q : R) := rfl

@[simp, norm_cast] lemma val_ratCast (q : ℚ) : (q : selfAdjoint R) = (q : R) := rfl


instance instSMulNNRat : SMul ℚ≥0 (selfAdjoint R) where
                               /-
                                 R : Type u_1
                                 A : Type u_2
                                 inst✝¹ : Field R
                                 inst✝ : StarRing R
                                 a : NNRat
                                 x : Subtype fun x => Membership.mem (selfAdjoint R) x
                                 ⊢ Membership.mem (selfAdjoint R) (HSMul.hSMul a ↑x)
                               -/
  smul a x := ⟨a • (x : R), by rw [NNRat.smul_def]; exact .mul (.nnratCast a) x.prop⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


instance instSMulRat : SMul ℚ (selfAdjoint R) where
                               /-
                                 R : Type u_1
                                 A : Type u_2
                                 inst✝¹ : Field R
                                 inst✝ : StarRing R
                                 a : Rat
                                 x : Subtype fun x => Membership.mem (selfAdjoint R) x
                                 ⊢ Membership.mem (selfAdjoint R) (HSMul.hSMul a ↑x)
                               -/
  smul a x := ⟨a • (x : R), by rw [Rat.smul_def]; exact .mul (.ratCast a) x.prop⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp, norm_cast] lemma val_nnqsmul (q : ℚ≥0) (x : selfAdjoint R) : ↑(q • x) = q • (x : R) := rfl

@[simp, norm_cast] lemma val_qsmul (q : ℚ) (x : selfAdjoint R) : ↑(q • x) = q • (x : R) := rfl


instance instField : Field (selfAdjoint R) :=
  Subtype.coe_injective.field _  (selfAdjoint R).coe_zero val_one
    (selfAdjoint R).coe_add val_mul (selfAdjoint R).coe_neg (selfAdjoint R).coe_sub
                                                         /-
                                                           R : Type u_1
                                                           A : Type u_2
                                                           inst✝¹ : Field R
                                                           inst✝ : StarRing R
                                                           ⊢ ∀ (n : Int) (x : Subtype fun x => Membership.mem (selfAdjoint R) x), Eq (↑(H …
                                                         -/
    val_inv val_div (swap (selfAdjoint R).coe_nsmul) (by intros; rfl) val_nnqsmul
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    val_qsmul val_pow val_zpow (fun _ => rfl) (fun _ => rfl) val_nnratCast val_ratCast


instance [SMul R A] [StarModule R A] : SMul R (selfAdjoint A) where
  smul r x := ⟨r • (x : A), (IsSelfAdjoint.all _).smul x.prop⟩


@[simp, norm_cast]
theorem val_smul [SMul R A] [StarModule R A] (r : R) (x : selfAdjoint A) : ↑(r • x) = r • (x : A) :=
  rfl


instance [Monoid R] [MulAction R A] [StarModule R A] : MulAction R (selfAdjoint A) :=
  Function.Injective.mulAction Subtype.val Subtype.coe_injective val_smul


instance [Monoid R] [DistribMulAction R A] [StarModule R A] : DistribMulAction R (selfAdjoint A) :=
  Function.Injective.distribMulAction (selfAdjoint A).subtype Subtype.coe_injective val_smul


instance [Semiring R] [Module R A] [StarModule R A] : Module R (selfAdjoint A) :=
  Function.Injective.module R (selfAdjoint A).subtype Subtype.coe_injective val_smul


theorem mem_iff {x : R} : x ∈ skewAdjoint R ↔ star x = -x := by
  /-
    R : Type u_1
    inst✝¹ : AddCommGroup R
    inst✝ : StarAddMonoid R
    x : R
    ⊢ Iff (Membership.mem (skewAdjoint R) x) (Eq (Star.star x) (Neg.neg x))
  -/
  rw [← AddSubgroup.mem_carrier]
  /-
    R : Type u_1
    inst✝¹ : AddCommGroup R
    inst✝ : StarAddMonoid R
    x : R
    ⊢ Iff (Membership.mem (skewAdjoint R).carrier x) (Eq (Star.star x) (Neg.neg x))
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem star_val_eq {x : skewAdjoint R} : star (x : R) = -x :=
  x.prop


instance : Inhabited (skewAdjoint R) :=
  ⟨0⟩


theorem conjugate {x : R} (hx : x ∈ skewAdjoint R) (z : R) : z * x * star z ∈ skewAdjoint R := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : StarRing R
    x : R
    hx : Membership.mem (skewAdjoint R) x
    z : R
    ⊢ Membership.mem (skewAdjoint R) (HMul.hMul (HMul.hMul z x) (Star.star z))
  -/
  simp only [mem_iff, star_mul, star_star, mem_iff.mp hx, neg_mul, mul_neg, mul_assoc]
  /-
    🎉 no goals
  -/


theorem conjugate' {x : R} (hx : x ∈ skewAdjoint R) (z : R) : star z * x * z ∈ skewAdjoint R := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : StarRing R
    x : R
    hx : Membership.mem (skewAdjoint R) x
    z : R
    ⊢ Membership.mem (skewAdjoint R) (HMul.hMul (HMul.hMul (Star.star z) x) z)
  -/
  simp only [mem_iff, star_mul, star_star, mem_iff.mp hx, neg_mul, mul_neg, mul_assoc]
  /-
    🎉 no goals
  -/


theorem isStarNormal_of_mem {x : R} (hx : x ∈ skewAdjoint R) : IsStarNormal x :=
  ⟨by
    /-
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : StarRing R
      x : R
      hx : Membership.mem (skewAdjoint R) x
      ⊢ Commute (Star.star x) x
    -/
    simp only [mem_iff] at hx
    /-
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : StarRing R
      x : R
      hx : Eq (Star.star x) (Neg.neg x)
      ⊢ Commute (Star.star x) x
    -/
    simp only [hx, Commute.neg_left, Commute.refl]⟩
    /-
      🎉 no goals
    -/


instance (x : skewAdjoint R) : IsStarNormal (x : R) :=
  isStarNormal_of_mem (SetLike.coe_mem _)


@[aesop safe apply (rule_sets := [SetLike])]
theorem smul_mem [Monoid R] [DistribMulAction R A] [StarModule R A] (r : R) {x : A}
    (h : x ∈ skewAdjoint A) : r • x ∈ skewAdjoint A := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁶ : Star R
    inst✝⁵ : TrivialStar R
    inst✝⁴ : AddCommGroup A
    inst✝³ : StarAddMonoid A
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R A
    inst✝ : StarModule R A
    r : R
    x : A
    h : Membership.mem (skewAdjoint A) x
    ⊢ Membership.mem (skewAdjoint A) (HSMul.hSMul r x)
  -/
  rw [mem_iff, star_smul, star_trivial, mem_iff.mp h, smul_neg r]
  /-
    🎉 no goals
  -/


instance [Monoid R] [DistribMulAction R A] [StarModule R A] : SMul R (skewAdjoint A) where
  smul r x := ⟨r • (x : A), smul_mem r x.prop⟩


@[simp, norm_cast]
theorem val_smul [Monoid R] [DistribMulAction R A] [StarModule R A] (r : R) (x : skewAdjoint A) :
    ↑(r • x) = r • (x : A) :=
  rfl


instance [Monoid R] [DistribMulAction R A] [StarModule R A] : DistribMulAction R (skewAdjoint A) :=
  Function.Injective.distribMulAction (skewAdjoint A).subtype Subtype.coe_injective val_smul


instance [Semiring R] [Module R A] [StarModule R A] : Module R (skewAdjoint A) :=
  Function.Injective.module R (skewAdjoint A).subtype Subtype.coe_injective val_smul


/-- Scalar multiplication of a self-adjoint element by a skew-adjoint element produces a
skew-adjoint element. -/
theorem IsSelfAdjoint.smul_mem_skewAdjoint [Ring R] [AddCommGroup A] [Module R A] [StarAddMonoid R]
    [StarAddMonoid A] [StarModule R A] {r : R} (hr : r ∈ skewAdjoint R) {a : A}
    (ha : IsSelfAdjoint a) : r • a ∈ skewAdjoint A :=
  (star_smul _ _).trans <| (congr_arg₂ _ hr ha).trans <| neg_smul _ _


/-- Scalar multiplication of a skew-adjoint element by a skew-adjoint element produces a
self-adjoint element. -/
theorem isSelfAdjoint_smul_of_mem_skewAdjoint [Ring R] [AddCommGroup A] [Module R A]
    [StarAddMonoid R] [StarAddMonoid A] [StarModule R A] {r : R} (hr : r ∈ skewAdjoint R) {a : A}
    (ha : a ∈ skewAdjoint A) : IsSelfAdjoint (r • a) :=
  (star_smul _ _).trans <| (congr_arg₂ _ hr ha).trans <| neg_smul_neg _ _


instance isStarNormal_zero [Semiring R] [StarRing R] : IsStarNormal (0 : R) :=
      /-
        R : Type u_1
        A : Type u_2
        inst✝¹ : Semiring R
        inst✝ : StarRing R
        ⊢ Commute (Star.star 0) 0
      -/
  ⟨by simp only [Commute.refl, star_comm_self, star_zero]⟩
      /-
        🎉 no goals
      -/


instance isStarNormal_one [MulOneClass R] [StarMul R] : IsStarNormal (1 : R) :=
      /-
        R : Type u_1
        A : Type u_2
        inst✝¹ : MulOneClass R
        inst✝ : StarMul R
        ⊢ Commute (Star.star 1) 1
      -/
  ⟨by simp only [Commute.refl, star_comm_self, star_one]⟩
      /-
        🎉 no goals
      -/


protected instance IsStarNormal.star [Mul R] [StarMul R] {x : R} [IsStarNormal x] :
    IsStarNormal (star x) :=
                                                           /-
                                                             R : Type u_1
                                                             A : Type u_2
                                                             inst✝² : Mul R
                                                             inst✝¹ : StarMul R
                                                             x : R
                                                             inst✝ : IsStarNormal x
                                                             ⊢ Eq (HMul.hMul (Star.star (Star.star x)) (Star.star x)) (HMul.hMul (Star.star …
                                                           -/
  ⟨show star (star x) * star x = star x * star (star x) by rw [star_star, star_comm_self']⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


protected instance IsStarNormal.neg [Ring R] [StarAddMonoid R] {x : R} [IsStarNormal x] :
    IsStarNormal (-x) :=
                                           /-
                                             R : Type u_1
                                             A : Type u_2
                                             inst✝² : Ring R
                                             inst✝¹ : StarAddMonoid R
                                             x : R
                                             inst✝ : IsStarNormal x
                                             ⊢ Eq (HMul.hMul (Star.star (Neg.neg x)) (Neg.neg x)) (HMul.hMul (Neg.neg x) (S …
                                           -/
  ⟨show star (-x) * -x = -x * star (-x) by simp_rw [star_neg, neg_mul_neg, star_comm_self']⟩
                                           /-
                                             🎉 no goals
                                           -/


protected instance IsStarNormal.map {F R S : Type*} [Mul R] [Star R] [Mul S] [Star S]
    [FunLike F R S] [MulHomClass F R S] [StarHomClass F R S] (f : F) (r : R) [hr : IsStarNormal r] :
    IsStarNormal (f r) where
                       /-
                         R✝ : Type u_1
                         A : Type u_2
                         F : Type u_3
                         R : Type u_4
                         S : Type u_5
                         inst✝⁶ : Mul R
                         inst✝⁵ : Star R
                         inst✝⁴ : Mul S
                         inst✝³ : Star S
                         inst✝² : FunLike F R S
                         inst✝¹ : MulHomClass F R S
                         inst✝ : StarHomClass F R S
                         f : F
                         r : R
                         hr : IsStarNormal r
                         ⊢ Commute (Star.star (f r)) (f r)
                       -/
  star_comm_self := by simpa [map_star] using congr(f $(hr.star_comm_self))
                       /-
                         🎉 no goals
                       -/

-- see Note [lower instance priority]

instance (priority := 100) TrivialStar.isStarNormal [Mul R] [StarMul R] [TrivialStar R]
    {x : R} : IsStarNormal x :=
      /-
        R : Type u_1
        A : Type u_2
        inst✝² : Mul R
        inst✝¹ : StarMul R
        inst✝ : TrivialStar R
        x : R
        ⊢ Commute (Star.star x) x
      -/
  ⟨by rw [star_trivial]⟩
      /-
        🎉 no goals
      -/

-- see Note [lower instance priority]

instance (priority := 100) CommMonoid.isStarNormal [CommMonoid R] [StarMul R] {x : R} :
    IsStarNormal x :=
  ⟨mul_comm _ _⟩



protected lemma isSelfAdjoint : IsSelfAdjoint f ↔ ∀ i, IsSelfAdjoint (f i) := funext_iff


alias ⟨_root_.IsSelfAdjoint.apply, _⟩ := Pi.isSelfAdjoint


