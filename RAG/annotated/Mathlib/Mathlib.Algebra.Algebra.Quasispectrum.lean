/-- A type synonym for non-unital rings where an alternative monoid structure is introduced.
If `R` is a non-unital semiring, then `PreQuasiregular R` is equipped with the monoid structure
with binary operation `fun x y ↦ y + x + x * y` and identity `0`. Elements of `R` which are
invertible in this monoid satisfy the predicate `IsQuasiregular`. -/
structure PreQuasiregular (R : Type*) where
  /-- The value wrapped into a term of `PreQuasiregular`. -/
  val : R


/-- The identity map between `R` and `PreQuasiregular R`. -/
@[simps]
def equiv : R ≃ PreQuasiregular R where
  toFun := .mk
  invFun := PreQuasiregular.val
  left_inv _ := rfl
  right_inv _ := rfl


instance instOne : One (PreQuasiregular R) where
  one := equiv 0


@[simp]
lemma val_one : (1 : PreQuasiregular R).val = 0 := rfl


instance instMul : Mul (PreQuasiregular R) where
  mul x y := .mk (y.val + x.val + x.val * y.val)


@[simp]
lemma val_mul (x y : PreQuasiregular R) : (x * y).val = y.val + x.val + x.val * y.val := rfl


instance instMonoid : Monoid (PreQuasiregular R) where
  one := equiv 0
  mul x y := .mk (y.val + x.val + x.val * y.val)
                                          /-
                                            R : Type u_1
                                            inst✝ : NonUnitalSemiring R
                                            x✝ : PreQuasiregular R
                                            ⊢ Eq (PreQuasiregular.equiv.symm (HMul.hMul x✝ 1)) (PreQuasiregular.equiv.symm …
                                          -/
                                          /-
                                            R : Type u_1
                                            inst✝ : NonUnitalSemiring R
                                            x✝ : PreQuasiregular R
                                            ⊢ Eq (PreQuasiregular.equiv.symm (HMul.hMul 1 x✝)) (PreQuasiregular.equiv.symm …
                                          -/
                                                /-
                                                  R : Type u_1
                                                  inst✝ : NonUnitalSemiring R
                                                  x y z : PreQuasiregular R
                                                  ⊢ Eq (PreQuasiregular.equiv.symm (HMul.hMul (HMul.hMul x y) z)) (PreQuasiregul …
                                                -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  mul_one _ := equiv.symm.injective <| by simp [-EmbeddingLike.apply_eq_iff_eq]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  one_mul _ := equiv.symm.injective <| by simp [-EmbeddingLike.apply_eq_iff_eq]
  mul_assoc x y z := equiv.symm.injective <| by simp [mul_add, add_mul, mul_assoc]; abel


@[simp]
lemma inv_add_add_mul_eq_zero (u : (PreQuasiregular R)ˣ) :
    u⁻¹.val.val + u.val.val + u.val.val * u⁻¹.val.val = 0 := by
  /-
    R : Type u_1
    inst✝ : NonUnitalSemiring R
    u : Units (PreQuasiregular R)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑(Inv.inv u)).val (↑u).val) (HMul.hMul (↑u).val (↑ …
  -/
  simpa [-Units.mul_inv] using congr($(u.mul_inv).val)
  /-
    🎉 no goals
  -/


@[simp]
lemma add_inv_add_mul_eq_zero (u : (PreQuasiregular R)ˣ) :
    u.val.val + u⁻¹.val.val + u⁻¹.val.val * u.val.val = 0 := by
  /-
    R : Type u_1
    inst✝ : NonUnitalSemiring R
    u : Units (PreQuasiregular R)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑u).val (↑(Inv.inv u)).val) (HMul.hMul (↑(Inv.inv  …
  -/
  simpa [-Units.inv_mul] using congr($(u.inv_mul).val)
  /-
    🎉 no goals
  -/


variable (R A) in
/-- The subgroup of the units of `Unitization R A` whose scalar part is `1`. -/
def unitsFstOne : Subgroup (Unitization R A)ˣ where
  carrier := {x | x.val.fst = 1}
  one_mem' := rfl
                                                                   /-
                                                                     R : Type u_1
                                                                     A : Type u_2
                                                                     inst✝⁴ : CommSemiring R
                                                                     inst✝³ : NonUnitalSemiring A
                                                                     inst✝² : Module R A
                                                                     inst✝¹ : IsScalarTower R A A
                                                                     inst✝ : SMulCommClass R A A
                                                                     x y : Units (Unitization R A)
                                                                     hx : Eq (↑x).fst 1
                                                                     hy : Eq (↑y).fst 1
                                                                     ⊢ Membership.mem (setOf fun x => Eq (↑x).fst 1) (HMul.hMul x y)
                                                                   -/
  mul_mem' {x} {y} (hx : fst x.val = 1) (hy : fst y.val = 1) := by simp [hx, hy]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  inv_mem' {x} (hx : fst x.val = 1) := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      x : Units (Unitization R A)
      hx : Eq (↑x).fst 1
      ⊢ Membership.mem { carrier := setOf fun x => Eq (↑x).fst 1, mul_mem' := ⋯, one …
    -/
    simpa [-Units.mul_inv, hx] using congr(fstHom R A $(x.mul_inv))
    /-
      🎉 no goals
    -/


@[simp]
lemma mem_unitsFstOne {x : (Unitization R A)ˣ} : x ∈ unitsFstOne R A ↔ x.val.fst = 1 := Iff.rfl


@[simp]
lemma unitsFstOne_val_val_fst (x : (unitsFstOne R A)) : x.val.val.fst = 1 :=
  mem_unitsFstOne.mp x.property


@[simp]
lemma unitsFstOne_val_inv_val_fst (x : (unitsFstOne R A)) : x.val⁻¹.val.fst = 1 :=
  mem_unitsFstOne.mp x⁻¹.property


variable (R) in
/-- If `A` is a non-unital `R`-algebra, then the subgroup of units of `Unitization R A` whose
scalar part is `1 : R` (i.e., `Unitization.unitsFstOne`) is isomorphic to the group of units of
`PreQuasiregular A`. -/
@[simps]
def unitsFstOne_mulEquiv_quasiregular : unitsFstOne R A ≃* (PreQuasiregular A)ˣ where
  toFun x :=
    { val := equiv x.val.val.snd
      inv := equiv x⁻¹.val.val.snd
      val_inv := equiv.symm.injective <| by
        /-
          R : Type u_1
          A : Type u_2
          inst✝⁴ : CommSemiring R
          inst✝³ : NonUnitalSemiring A
          inst✝² : Module R A
          inst✝¹ : IsScalarTower R A A
          inst✝ : SMulCommClass R A A
          x : Subtype fun x => Membership.mem (Unitization.unitsFstOne R A) x
          ⊢ Eq (PreQuasiregular.equiv.symm (HMul.hMul (PreQuasiregular.equiv (↑↑x).snd)  …
        -/
        simpa [-Units.mul_inv] using congr(snd $(x.val.mul_inv))
        /-
          🎉 no goals
        -/
      inv_val := equiv.symm.injective <| by
        /-
          R : Type u_1
          A : Type u_2
          inst✝⁴ : CommSemiring R
          inst✝³ : NonUnitalSemiring A
          inst✝² : Module R A
          inst✝¹ : IsScalarTower R A A
          inst✝ : SMulCommClass R A A
          x : Subtype fun x => Membership.mem (Unitization.unitsFstOne R A) x
          ⊢ Eq (PreQuasiregular.equiv.symm (HMul.hMul (PreQuasiregular.equiv (↑↑(Inv.inv …
        -/
        simpa [-Units.inv_mul] using congr(snd $(x.val.inv_mul)) }
        /-
          🎉 no goals
        -/
  invFun x :=
    { val :=
      { val := 1 + equiv.symm x.val
        inv := 1 + equiv.symm x⁻¹.val
        val_inv := by
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁴ : CommSemiring R
            inst✝³ : NonUnitalSemiring A
            inst✝² : Module R A
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            x : Units (PreQuasiregular A)
            ⊢ Eq (HMul.hMul (HAdd.hAdd 1 ↑(PreQuasiregular.equiv.symm ↑x)) (HAdd.hAdd 1 ↑( …
          -/
          convert congr((1 + $(inv_add_add_mul_eq_zero x) : Unitization R A)) using 1
          · simp only [mul_one, equiv_symm_apply, one_mul, inr_zero, add_zero, mul_add, add_mul,
              inr_add, inr_mul]
            /-
              case h.e'_2
              R : Type u_1
              A : Type u_2
              inst✝⁴ : CommSemiring R
              inst✝³ : NonUnitalSemiring A
              inst✝² : Module R A
              inst✝¹ : IsScalarTower R A A
              inst✝ : SMulCommClass R A A
              x : Units (PreQuasiregular A)
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 ↑(↑x).val) (HAdd.hAdd (↑(↑(Inv.inv x)).val) (HMul …
            -/
            /-
              🎉 no goals
            -/
            abel
            /-
              🎉 no goals
            -/
            /-
              case h.e'_3
              R : Type u_1
              A : Type u_2
              inst✝⁴ : CommSemiring R
              inst✝³ : NonUnitalSemiring A
              inst✝² : Module R A
              inst✝¹ : IsScalarTower R A A
              inst✝ : SMulCommClass R A A
              x : Units (PreQuasiregular A)
              ⊢ Eq 1 (HAdd.hAdd 1 ↑0)
            -/
          · simp only [inr_zero, add_zero]
            /-
              🎉 no goals
            -/
        inv_val := by
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁴ : CommSemiring R
            inst✝³ : NonUnitalSemiring A
            inst✝² : Module R A
            inst✝¹ : IsScalarTower R A A
            inst✝ : SMulCommClass R A A
            x : Units (PreQuasiregular A)
            ⊢ Eq (HMul.hMul (HAdd.hAdd 1 ↑(PreQuasiregular.equiv.symm ↑(Inv.inv x))) (HAdd …
          -/
          convert congr((1 + $(add_inv_add_mul_eq_zero x) : Unitization R A)) using 1
          · simp only [mul_one, equiv_symm_apply, one_mul, inr_zero, add_zero, mul_add, add_mul,
              inr_add, inr_mul]
            /-
              case h.e'_2
              R : Type u_1
              A : Type u_2
              inst✝⁴ : CommSemiring R
              inst✝³ : NonUnitalSemiring A
              inst✝² : Module R A
              inst✝¹ : IsScalarTower R A A
              inst✝ : SMulCommClass R A A
              x : Units (PreQuasiregular A)
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd 1 ↑(↑(Inv.inv x)).val) (HAdd.hAdd (↑(↑x).val) (HMul …
            -/
            /-
              🎉 no goals
            -/
            abel
            /-
              🎉 no goals
            -/
            /-
              case h.e'_3
              R : Type u_1
              A : Type u_2
              inst✝⁴ : CommSemiring R
              inst✝³ : NonUnitalSemiring A
              inst✝² : Module R A
              inst✝¹ : IsScalarTower R A A
              inst✝ : SMulCommClass R A A
              x : Units (PreQuasiregular A)
              ⊢ Eq 1 (HAdd.hAdd 1 ↑0)
            -/
          · simp only [inr_zero, add_zero] }
            /-
              🎉 no goals
            -/
                     /-
                       R : Type u_1
                       A : Type u_2
                       inst✝⁴ : CommSemiring R
                       inst✝³ : NonUnitalSemiring A
                       inst✝² : Module R A
                       inst✝¹ : IsScalarTower R A A
                       inst✝ : SMulCommClass R A A
                       x : Units (PreQuasiregular A)
                       ⊢ Membership.mem (Unitization.unitsFstOne R A) { val := HAdd.hAdd 1 ↑(PreQuasi …
                     -/
      property := by simp }
                     /-
                       🎉 no goals
                     -/
                                               /-
                                                 R : Type u_1
                                                 A : Type u_2
                                                 inst✝⁴ : CommSemiring R
                                                 inst✝³ : NonUnitalSemiring A
                                                 inst✝² : Module R A
                                                 inst✝¹ : IsScalarTower R A A
                                                 inst✝ : SMulCommClass R A A
                                                 x : Subtype fun x => Membership.mem (Unitization.unitsFstOne R A) x
                                                 ⊢ Eq ↑↑((fun x => ⟨{ val := HAdd.hAdd 1 ↑(PreQuasiregular.equiv.symm ↑x), inv  …
                                               -/
  left_inv x := Subtype.ext <| Units.ext <| by simpa using x.val.val.inl_fst_add_inr_snd_eq
                                               /-
                                                 🎉 no goals
                                               -/
                                 /-
                                   R : Type u_1
                                   A : Type u_2
                                   inst✝⁴ : CommSemiring R
                                   inst✝³ : NonUnitalSemiring A
                                   inst✝² : Module R A
                                   inst✝¹ : IsScalarTower R A A
                                   inst✝ : SMulCommClass R A A
                                   x : Units (PreQuasiregular A)
                                   ⊢ Eq ↑((fun x => { val := PreQuasiregular.equiv (↑↑x).snd, inv := PreQuasiregu …
                                 -/
  right_inv x := Units.ext <| by simp [-equiv_symm_apply]
                                 /-
                                   🎉 no goals
                                 -/
                                                          /-
                                                            R : Type u_1
                                                            A : Type u_2
                                                            inst✝⁴ : CommSemiring R
                                                            inst✝³ : NonUnitalSemiring A
                                                            inst✝² : Module R A
                                                            inst✝¹ : IsScalarTower R A A
                                                            inst✝ : SMulCommClass R A A
                                                            x y : Subtype fun x => Membership.mem (Unitization.unitsFstOne R A) x
                                                            ⊢ Eq (PreQuasiregular.equiv.symm ↑({ toFun := fun x => { val := PreQuasiregula …
                                                          -/
  map_mul' x y := Units.ext <| equiv.symm.injective <| by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- In a non-unital semiring `R`, an element `x : R` satisfies `IsQuasiregular` if it is a unit
under the monoid operation `fun x y ↦ y + x + x * y`. -/
def IsQuasiregular (x : R) : Prop :=
  ∃ u : (PreQuasiregular R)ˣ, equiv.symm u.val = x


@[simp]
lemma isQuasiregular_zero : IsQuasiregular 0 := ⟨1, rfl⟩


lemma isQuasiregular_iff {x : R} :
    IsQuasiregular x ↔ ∃ y, y + x + x * y = 0 ∧ x + y + y * x = 0 := by
  /-
    R : Type u_1
    inst✝ : NonUnitalSemiring R
    x : R
    ⊢ Iff (IsQuasiregular x) (Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y x) ( …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : NonUnitalSemiring R
      x : R
      ⊢ IsQuasiregular x → Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul. …
    -/
  · rintro ⟨u, rfl⟩
    /-
      case mp.intro
      R : Type u_1
      inst✝ : NonUnitalSemiring R
      u : Units (PreQuasiregular R)
      ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y (PreQuasiregular.equiv.symm  …
    -/
    exact ⟨equiv.symm u⁻¹.val, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : NonUnitalSemiring R
      x : R
      ⊢ (Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0) (Eq  …
    -/
  · rintro ⟨y, hy₁, hy₂⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝ : NonUnitalSemiring R
      x y : R
      hy₁ : Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0
      hy₂ : Eq (HAdd.hAdd (HAdd.hAdd x y) (HMul.hMul y x)) 0
      ⊢ IsQuasiregular x
    -/
    refine ⟨⟨equiv x, equiv y, ?_, ?_⟩, rfl⟩
    all_goals
      apply equiv.symm.injective
      assumption


lemma IsQuasiregular.map {F R S : Type*} [NonUnitalSemiring R] [NonUnitalSemiring S]
    [FunLike F R S] [NonUnitalRingHomClass F R S] (f : F) {x : R} (hx : IsQuasiregular x) :
    IsQuasiregular (f x) := by
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝³ : NonUnitalSemiring R
    inst✝² : NonUnitalSemiring S
    inst✝¹ : FunLike F R S
    inst✝ : NonUnitalRingHomClass F R S
    f : F
    x : R
    hx : IsQuasiregular x
    ⊢ IsQuasiregular (f x)
  -/
  rw [isQuasiregular_iff] at hx ⊢
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝³ : NonUnitalSemiring R
    inst✝² : NonUnitalSemiring S
    inst✝¹ : FunLike F R S
    inst✝ : NonUnitalRingHomClass F R S
    f : F
    x : R
    hx : Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0) (E …
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y (f x)) (HMul.hMul (f x) y))  …
  -/
  obtain ⟨y, hy₁, hy₂⟩ := hx
  /-
    case intro.intro
    F : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝³ : NonUnitalSemiring R
    inst✝² : NonUnitalSemiring S
    inst✝¹ : FunLike F R S
    inst✝ : NonUnitalRingHomClass F R S
    f : F
    x y : R
    hy₁ : Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0
    hy₂ : Eq (HAdd.hAdd (HAdd.hAdd x y) (HMul.hMul y x)) 0
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y (f x)) (HMul.hMul (f x) y))  …
  -/
  exact ⟨f y, by simpa using And.intro congr(f $(hy₁)) congr(f $(hy₂))⟩
  /-
    🎉 no goals
  -/


lemma IsQuasiregular.isUnit_one_add {R : Type*} [Semiring R] {x : R} (hx : IsQuasiregular x) :
    IsUnit (1 + x) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : R
    hx : IsQuasiregular x
    ⊢ IsUnit (HAdd.hAdd 1 x)
  -/
  obtain ⟨y, hy₁, hy₂⟩ := isQuasiregular_iff.mp hx
  /-
    case intro.intro
    R : Type u_1
    inst✝ : Semiring R
    x : R
    hx : IsQuasiregular x
    y : R
    hy₁ : Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0
    hy₂ : Eq (HAdd.hAdd (HAdd.hAdd x y) (HMul.hMul y x)) 0
    ⊢ IsUnit (HAdd.hAdd 1 x)
  -/
  refine ⟨⟨1 + x, 1 + y, ?_, ?_⟩, rfl⟩
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝ : Semiring R
      x : R
      hx : IsQuasiregular x
      y : R
      hy₁ : Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0
      hy₂ : Eq (HAdd.hAdd (HAdd.hAdd x y) (HMul.hMul y x)) 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd 1 x) (HAdd.hAdd 1 y)) 1
    -/
  · convert congr(1 + $(hy₁)) using 1 <;> [noncomm_ring; simp]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝ : Semiring R
      x : R
      hx : IsQuasiregular x
      y : R
      hy₁ : Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0
      hy₂ : Eq (HAdd.hAdd (HAdd.hAdd x y) (HMul.hMul y x)) 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd 1 y) (HAdd.hAdd 1 x)) 1
    -/
  · convert congr(1 + $(hy₂)) using 1 <;> [noncomm_ring; simp]
    /-
      🎉 no goals
    -/


lemma isQuasiregular_iff_isUnit {R : Type*} [Ring R] {x : R} :
    IsQuasiregular x ↔ IsUnit (1 + x) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    ⊢ Iff (IsQuasiregular x) (IsUnit (HAdd.hAdd 1 x))
  -/
  refine ⟨IsQuasiregular.isUnit_one_add, fun hx ↦ ?_⟩
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    hx : IsUnit (HAdd.hAdd 1 x)
    ⊢ IsQuasiregular x
  -/
  rw [isQuasiregular_iff]
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    hx : IsUnit (HAdd.hAdd 1 x)
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y x) (HMul.hMul x y)) 0) (Eq ( …
  -/
  use hx.unit⁻¹ - 1
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    x : R
    hx : IsUnit (HAdd.hAdd 1 x)
    ⊢ And (Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (↑(Inv.inv hx.unit)) 1) x) (HMul.hM …
  -/
  constructor
  /-
    case h.left
    R : Type u_1
    inst✝ : Ring R
    x : R
    hx : IsUnit (HAdd.hAdd 1 x)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (↑(Inv.inv hx.unit)) 1) x) (HMul.hMul x  …
  -/
  case' h.left => have := congr($(hx.mul_val_inv) - 1)
  /-
    case h.left
    R : Type u_1
    inst✝ : Ring R
    x : R
    hx : IsUnit (HAdd.hAdd 1 x)
    this : Eq (HSub.hSub (HMul.hMul (HAdd.hAdd 1 x) ↑(Inv.inv hx.unit)) 1) (HSub.h …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSub.hSub (↑(Inv.inv hx.unit)) 1) x) (HMul.hMul x  …
  -/
  case' h.right => have := congr($(hx.val_inv_mul) - 1)
  all_goals
    rw [← sub_add_cancel (↑hx.unit⁻¹ : R) 1, sub_self] at this
    convert this using 1
    noncomm_ring

-- interestingly, this holds even in the semiring case.

lemma isQuasiregular_iff_isUnit' (R : Type*) {A : Type*} [CommSemiring R] [NonUnitalSemiring A]
    [Module R A] [IsScalarTower R A A] [SMulCommClass R A A] {x : A} :
    IsQuasiregular x ↔ IsUnit (1 + x : Unitization R A) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalSemiring A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    x : A
    ⊢ Iff (IsQuasiregular x) (IsUnit (HAdd.hAdd 1 ↑x))
  -/
  refine ⟨?_, fun hx ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      x : A
      ⊢ IsQuasiregular x → IsUnit (HAdd.hAdd 1 ↑x)
    -/
  · rintro ⟨u, rfl⟩
    /-
      case refine_1.intro
      R : Type u_1
      A : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      u : Units (PreQuasiregular A)
      ⊢ IsUnit (HAdd.hAdd 1 ↑(PreQuasiregular.equiv.symm ↑u))
    -/
    exact (Unitization.unitsFstOne_mulEquiv_quasiregular R).symm u |>.val.isUnit
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : NonUnitalSemiring A
      inst✝² : Module R A
      inst✝¹ : IsScalarTower R A A
      inst✝ : SMulCommClass R A A
      x : A
      hx : IsUnit (HAdd.hAdd 1 ↑x)
      ⊢ IsQuasiregular x
    -/
  · exact ⟨(Unitization.unitsFstOne_mulEquiv_quasiregular R) ⟨hx.unit, by simp⟩, by simp⟩
    /-
      🎉 no goals
    -/


/-- If `A` is a non-unital `R`-algebra, the `R`-quasispectrum of `a : A` consists of those `r : R`
such that if `r` is invertible (in `R`), then `-(r⁻¹ • a)` is not quasiregular.

The quasispectrum is precisely the spectrum in the unitization when `R` is a commutative ring.
See `Unitization.quasispectrum_eq_spectrum_inr`. -/
def quasispectrum (a : A) : Set R :=
  {r : R | (hr : IsUnit r) → ¬ IsQuasiregular (-(hr.unit⁻¹ • a))}


variable {R} in
lemma quasispectrum.not_isUnit_mem (a : A) {r : R} (hr : ¬ IsUnit r) : r ∈ quasispectrum R a :=
  fun hr' ↦ (hr hr').elim


@[simp]
lemma quasispectrum.zero_mem [Nontrivial R] (a : A) : 0 ∈ quasispectrum R a :=
                                       /-
                                         R : Type u_1
                                         A : Type u_2
                                         inst✝³ : CommSemiring R
                                         inst✝² : NonUnitalRing A
                                         inst✝¹ : Module R A
                                         inst✝ : Nontrivial R
                                         a : A
                                         ⊢ Not (IsUnit 0)
                                       -/
  quasispectrum.not_isUnit_mem a <| by simp
                                       /-
                                         🎉 no goals
                                       -/


theorem quasispectrum.nonempty [Nontrivial R] (a : A) : (quasispectrum R a).Nonempty :=
  Set.nonempty_of_mem <| quasispectrum.zero_mem R a


instance quasispectrum.instZero [Nontrivial R] (a : A) : Zero (quasispectrum R a) where
  zero := ⟨0, quasispectrum.zero_mem R a⟩


/-- A version of `NonUnitalAlgHom.quasispectrum_apply_subset` which allows for `quasispectrum R`,
where `R` is a *semi*ring, but `φ` must still function over a scalar ring `S`. In this case, we
need `S` to be explicit. The primary use case is, for instance, `R := ℝ≥0` and `S := ℝ` or
`S := ℂ`. -/
lemma NonUnitalAlgHom.quasispectrum_apply_subset' {F R : Type*} (S : Type*) {A B : Type*}
    [CommSemiring R] [CommRing S] [NonUnitalRing A] [NonUnitalRing B] [Module R S]
    [Module S A] [Module R A] [Module S B] [Module R B] [IsScalarTower R S A] [IsScalarTower R S B]
    [FunLike F A B] [NonUnitalAlgHomClass F S A B] (φ : F) (a : A) :
    quasispectrum R (φ a) ⊆ quasispectrum R a := by
  /-
    F : Type u_3
    R : Type u_4
    S : Type u_5
    A : Type u_6
    B : Type u_7
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : NonUnitalRing B
    inst✝⁸ : Module R S
    inst✝⁷ : Module S A
    inst✝⁶ : Module R A
    inst✝⁵ : Module S B
    inst✝⁴ : Module R B
    inst✝³ : IsScalarTower R S A
    inst✝² : IsScalarTower R S B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F S A B
    φ : F
    a : A
    ⊢ HasSubset.Subset (quasispectrum R (φ a)) (quasispectrum R a)
  -/
  refine Set.compl_subset_compl.mp fun x ↦ ?_
  simp only [quasispectrum, Set.mem_compl_iff, Set.mem_setOf_eq, not_forall, not_not,
    forall_exists_index]
  /-
    F : Type u_3
    R : Type u_4
    S : Type u_5
    A : Type u_6
    B : Type u_7
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : NonUnitalRing B
    inst✝⁸ : Module R S
    inst✝⁷ : Module S A
    inst✝⁶ : Module R A
    inst✝⁵ : Module S B
    inst✝⁴ : Module R B
    inst✝³ : IsScalarTower R S A
    inst✝² : IsScalarTower R S B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F S A B
    φ : F
    a : A
    x : R
    ⊢ ∀ (x_1 : IsUnit x), IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv ⋯.unit) a) …
  -/
  refine fun hx this ↦ ⟨hx, ?_⟩
  /-
    F : Type u_3
    R : Type u_4
    S : Type u_5
    A : Type u_6
    B : Type u_7
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : NonUnitalRing B
    inst✝⁸ : Module R S
    inst✝⁷ : Module S A
    inst✝⁶ : Module R A
    inst✝⁵ : Module S B
    inst✝⁴ : Module R B
    inst✝³ : IsScalarTower R S A
    inst✝² : IsScalarTower R S B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F S A B
    φ : F
    a : A
    x : R
    hx : IsUnit x
    this : IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv ⋯.unit) a))
    ⊢ IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv ⋯.unit) (φ a)))
  -/
  rw [Units.smul_def, ← smul_one_smul S] at this ⊢
  /-
    F : Type u_3
    R : Type u_4
    S : Type u_5
    A : Type u_6
    B : Type u_7
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : NonUnitalRing A
    inst✝⁹ : NonUnitalRing B
    inst✝⁸ : Module R S
    inst✝⁷ : Module S A
    inst✝⁶ : Module R A
    inst✝⁵ : Module S B
    inst✝⁴ : Module R B
    inst✝³ : IsScalarTower R S A
    inst✝² : IsScalarTower R S B
    inst✝¹ : FunLike F A B
    inst✝ : NonUnitalAlgHomClass F S A B
    φ : F
    a : A
    x : R
    hx : IsUnit x
    this : IsQuasiregular (Neg.neg (HSMul.hSMul (HSMul.hSMul (↑(Inv.inv ⋯.unit)) 1 …
    ⊢ IsQuasiregular (Neg.neg (HSMul.hSMul (HSMul.hSMul (↑(Inv.inv ⋯.unit)) 1) (φ  …
  -/
  simpa [- smul_assoc] using this.map φ
  /-
    🎉 no goals
  -/


/-- If `φ` is non-unital algebra homomorphism over a scalar ring `R`, then
`quasispectrum R (φ a) ⊆ quasispectrum R a`. -/
lemma NonUnitalAlgHom.quasispectrum_apply_subset {F R A B : Type*}
    [CommRing R] [NonUnitalRing A] [NonUnitalRing B] [Module R A] [Module R B]
    [FunLike F A B] [NonUnitalAlgHomClass F R A B] (φ : F) (a : A) :
    quasispectrum R (φ a) ⊆ quasispectrum R a :=
  NonUnitalAlgHom.quasispectrum_apply_subset' R φ a


@[simp]
lemma quasispectrum.coe_zero [Nontrivial R] (a : A) : (0 : quasispectrum R a) = (0 : R) := rfl


lemma quasispectrum.mem_of_not_quasiregular (a : A) {r : Rˣ}
    (hr : ¬ IsQuasiregular (-(r⁻¹ • a))) : (r : R) ∈ quasispectrum R a :=
             /-
               R : Type u_1
               A : Type u_2
               inst✝² : CommSemiring R
               inst✝¹ : NonUnitalRing A
               inst✝ : Module R A
               a : A
               r : Units R
               hr : Not (IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv r) a)))
               x✝ : IsUnit ↑r
               ⊢ Not (IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv x✝.unit) a)))
             -/
  fun _ ↦ by simpa using hr
             /-
               🎉 no goals
             -/


lemma quasispectrum_eq_spectrum_union (R : Type*) {A : Type*} [CommSemiring R]
    [Ring A] [Algebra R A] (a : A) : quasispectrum R a = spectrum R a ∪ {r : R | ¬ IsUnit r} := by
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    ⊢ Eq (quasispectrum R a) (Union.union (spectrum R a) (setOf fun r => Not (IsUn …
  -/
  ext r
  /-
    case h
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (Union.union (spe …
  -/
  rw [quasispectrum]
  /-
    case h
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    ⊢ Iff (Membership.mem (setOf fun r => ∀ (hr : IsUnit r), Not (IsQuasiregular ( …
  -/
  simp only [Set.mem_setOf_eq, Set.mem_union, ← imp_iff_or_not, spectrum.mem_iff]
  /-
    case h
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    ⊢ Iff (∀ (hr : IsUnit r), Not (IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv h …
  -/
  congr! 1 with hr
  /-
    case h.a.h.a
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    hr : IsUnit r
    ⊢ Iff (Not (IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv hr.unit) a)))) (Not  …
  -/
  rw [not_iff_not, isQuasiregular_iff_isUnit, ← sub_eq_add_neg, Algebra.algebraMap_eq_smul_one]
  /-
    case h.a.h.a
    R : Type u_3
    A : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    hr : IsUnit r
    ⊢ Iff (IsUnit (HSub.hSub 1 (HSMul.hSMul (Inv.inv hr.unit) a))) (IsUnit (HSub.h …
  -/
  exact (IsUnit.smul_sub_iff_sub_inv_smul hr.unit a).symm
  /-
    🎉 no goals
  -/


lemma spectrum_subset_quasispectrum (R : Type*) {A : Type*} [CommSemiring R] [Ring A] [Algebra R A]
    (a : A) : spectrum R a ⊆ quasispectrum R a :=
  quasispectrum_eq_spectrum_union R a ▸ Set.subset_union_left


lemma quasispectrum_eq_spectrum_union_zero (R : Type*) {A : Type*} [Semifield R] [Ring A]
    [Algebra R A] (a : A) : quasispectrum R a = spectrum R a ∪ {0} := by
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : Semifield R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    ⊢ Eq (quasispectrum R a) (Union.union (spectrum R a) (Singleton.singleton 0))
  -/
  convert quasispectrum_eq_spectrum_union R a
  /-
    case h.e'_3.h.e'_4
    R : Type u_3
    A : Type u_4
    inst✝² : Semifield R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    ⊢ Eq (Singleton.singleton 0) (setOf fun r => Not (IsUnit r))
  -/
  ext x
  /-
    case h.e'_3.h.e'_4.h
    R : Type u_3
    A : Type u_4
    inst✝² : Semifield R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    x : R
    ⊢ Iff (Membership.mem (Singleton.singleton 0) x) (Membership.mem (setOf fun r  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma mem_quasispectrum_iff {R A : Type*} [Semifield R] [Ring A]
    [Algebra R A] {a : A} {x : R} :
    x ∈ quasispectrum R a ↔ x = 0 ∨ x ∈ spectrum R a := by
  /-
    R : Type u_3
    A : Type u_4
    inst✝² : Semifield R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    x : R
    ⊢ Iff (Membership.mem (quasispectrum R a) x) (Or (Eq x 0) (Membership.mem (spe …
  -/
  simp [quasispectrum_eq_spectrum_union_zero]
  /-
    🎉 no goals
  -/


lemma isQuasiregular_inr_iff (a : A) :
    IsQuasiregular (a : Unitization R A) ↔ IsQuasiregular a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    ⊢ Iff (IsQuasiregular ↑a) (IsQuasiregular a)
  -/
  refine ⟨fun ha ↦ ?_, IsQuasiregular.map (inrNonUnitalAlgHom R A)⟩
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    ha : IsQuasiregular ↑a
    ⊢ IsQuasiregular a
  -/
  rw [isQuasiregular_iff] at ha ⊢
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    ha : Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y ↑a) (HMul.hMul (↑a) y)) 0 …
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y a) (HMul.hMul a y)) 0) (Eq ( …
  -/
  obtain ⟨y, hy₁, hy₂⟩ := ha
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    y : Unitization R A
    hy₁ : Eq (HAdd.hAdd (HAdd.hAdd y ↑a) (HMul.hMul (↑a) y)) 0
    hy₂ : Eq (HAdd.hAdd (HAdd.hAdd (↑a) y) (HMul.hMul y ↑a)) 0
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y a) (HMul.hMul a y)) 0) (Eq ( …
  -/
  lift y to A using by simpa using congr(fstHom R A $(hy₁))
  /-
    case intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a y : A
    hy₁ : Eq (HAdd.hAdd (HAdd.hAdd ↑y ↑a) (HMul.hMul ↑a ↑y)) 0
    hy₂ : Eq (HAdd.hAdd (HAdd.hAdd ↑a ↑y) (HMul.hMul ↑y ↑a)) 0
    ⊢ Exists fun y => And (Eq (HAdd.hAdd (HAdd.hAdd y a) (HMul.hMul a y)) 0) (Eq ( …
  -/
                         /-
                           🎉 no goals
                         -/
  refine ⟨y, ?_, ?_⟩ <;> exact inr_injective (R := R) <| by simpa
                         /-
                           🎉 no goals
                         -/


lemma zero_mem_spectrum_inr (R S : Type*) {A : Type*} [CommSemiring R]
    [CommRing S] [Nontrivial S] [NonUnitalRing A] [Algebra R S] [Module S A] [IsScalarTower S A A]
    [SMulCommClass S A A] [Module R A] [IsScalarTower R S A] (a : A) :
    0 ∈ spectrum R (a : Unitization S A) := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁹ : CommSemiring R
    inst✝⁸ : CommRing S
    inst✝⁷ : Nontrivial S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    ⊢ Membership.mem (spectrum R ↑a) 0
  -/
  rw [spectrum.zero_mem_iff]
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁹ : CommSemiring R
    inst✝⁸ : CommRing S
    inst✝⁷ : Nontrivial S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    ⊢ Not (IsUnit ↑a)
  -/
  rintro ⟨u, hu⟩
  /-
    case intro
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁹ : CommSemiring R
    inst✝⁸ : CommRing S
    inst✝⁷ : Nontrivial S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    u : Units (Unitization S A)
    hu : Eq ↑u ↑a
    ⊢ False
  -/
  simpa [-Units.mul_inv, hu] using congr($(u.mul_inv).fst)
  /-
    🎉 no goals
  -/


lemma mem_spectrum_inr_of_not_isUnit {R A : Type*} [CommRing R]
    [NonUnitalRing A] [Module R A] [IsScalarTower R A A] [SMulCommClass R A A]
    (a : A) (r : R) (hr : ¬ IsUnit r) : r ∈ spectrum R (a : Unitization R A) :=
                   /-
                     R : Type u_3
                     A : Type u_4
                     inst✝⁴ : CommRing R
                     inst✝³ : NonUnitalRing A
                     inst✝² : Module R A
                     inst✝¹ : IsScalarTower R A A
                     inst✝ : SMulCommClass R A A
                     a : A
                     r : R
                     hr : Not (IsUnit r)
                     h : Membership.mem (resolventSet R ↑a) r
                     ⊢ IsUnit r
                   -/
  fun h ↦ hr <| by simpa [map_sub] using h.map (fstHom R A)
                   /-
                     🎉 no goals
                   -/


lemma quasispectrum_eq_spectrum_inr (R : Type*) {A : Type*} [CommRing R] [NonUnitalRing A]
    [Module R A] [IsScalarTower R A A] [SMulCommClass R A A] (a : A) :
    quasispectrum R a = spectrum R (a : Unitization R A) := by
  /-
    R : Type u_3
    A : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    ⊢ Eq (quasispectrum R a) (spectrum R ↑a)
  -/
  ext r
  /-
    case h
    R : Type u_3
    A : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    r : R
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (spectrum R ↑a) r)
  -/
  have : { r | ¬ IsUnit r} ⊆ spectrum R _ := mem_spectrum_inr_of_not_isUnit a
  /-
    case h
    R : Type u_3
    A : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    r : R
    this : HasSubset.Subset (setOf fun r => Not (IsUnit r)) (spectrum R ↑a)
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (spectrum R ↑a) r)
  -/
  rw [← Set.union_eq_left.mpr this, ← quasispectrum_eq_spectrum_union]
  /-
    case h
    R : Type u_3
    A : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    r : R
    this : HasSubset.Subset (setOf fun r => Not (IsUnit r)) (spectrum R ↑a)
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (quasispectrum R  …
  -/
  apply forall_congr' fun hr ↦ ?_
  /-
    R : Type u_3
    A : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : IsScalarTower R A A
    inst✝ : SMulCommClass R A A
    a : A
    r : R
    this : HasSubset.Subset (setOf fun r => Not (IsUnit r)) (spectrum R ↑a)
    hr : IsUnit r
    ⊢ Iff (Not (IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv hr.unit) a)))) (Not  …
  -/
  rw [not_iff_not, Units.smul_def, Units.smul_def, ← inr_smul, ← inr_neg, isQuasiregular_inr_iff]
  /-
    🎉 no goals
  -/


lemma quasispectrum_eq_spectrum_inr' (R S : Type*) {A : Type*} [Semifield R]
    [Field S] [NonUnitalRing A] [Algebra R S] [Module S A] [IsScalarTower S A A]
    [SMulCommClass S A A] [Module R A] [IsScalarTower R S A] (a : A) :
    quasispectrum R a = spectrum R (a : Unitization S A) := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    ⊢ Eq (quasispectrum R a) (spectrum R ↑a)
  -/
  ext r
  /-
    case h
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    r : R
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (spectrum R ↑a) r)
  -/
  have := Set.singleton_subset_iff.mpr (zero_mem_spectrum_inr R S a)
  /-
    case h
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    r : R
    this : HasSubset.Subset (Singleton.singleton 0) (spectrum R ↑a)
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (spectrum R ↑a) r)
  -/
  rw [← Set.union_eq_self_of_subset_right this, ← quasispectrum_eq_spectrum_union_zero]
  /-
    case h
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    r : R
    this : HasSubset.Subset (Singleton.singleton 0) (spectrum R ↑a)
    ⊢ Iff (Membership.mem (quasispectrum R a) r) (Membership.mem (quasispectrum R  …
  -/
  apply forall_congr' fun x ↦ ?_
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    r : R
    this : HasSubset.Subset (Singleton.singleton 0) (spectrum R ↑a)
    x : IsUnit r
    ⊢ Iff (Not (IsQuasiregular (Neg.neg (HSMul.hSMul (Inv.inv x.unit) a)))) (Not ( …
  -/
  rw [not_iff_not, Units.smul_def, Units.smul_def, ← inr_smul, ← inr_neg, isQuasiregular_inr_iff]
  /-
    🎉 no goals
  -/


lemma quasispectrum_inr_eq (R S : Type*) {A : Type*} [Semifield R]
    [Field S] [NonUnitalRing A] [Algebra R S] [Module S A] [IsScalarTower S A A]
    [SMulCommClass S A A] [Module R A] [IsScalarTower R S A] (a : A) :
    quasispectrum R (a : Unitization S A) = quasispectrum R a := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    ⊢ Eq (quasispectrum R ↑a) (quasispectrum R a)
  -/
  rw [quasispectrum_eq_spectrum_union_zero, quasispectrum_eq_spectrum_inr' R S]
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    ⊢ Eq (Union.union (spectrum R ↑a) (Singleton.singleton 0)) (spectrum R ↑a)
  -/
  apply Set.union_eq_self_of_subset_right
  /-
    case h
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    ⊢ HasSubset.Subset (Singleton.singleton 0) (spectrum R ↑a)
  -/
  simpa using zero_mem_spectrum_inr _ _ _
  /-
    🎉 no goals
  -/


/-- A class for `𝕜`-algebras with a partial order where the ordering is compatible with the
(quasi)spectrum. -/
class NonnegSpectrumClass (𝕜 A : Type*) [OrderedCommSemiring 𝕜] [NonUnitalRing A] [PartialOrder A]
    [Module 𝕜 A] : Prop where
  quasispectrum_nonneg_of_nonneg : ∀ a : A, 0 ≤ a → ∀ x ∈ quasispectrum 𝕜 a, 0 ≤ x


lemma iff_spectrum_nonneg {𝕜 A : Type*} [LinearOrderedSemifield 𝕜] [Ring A] [PartialOrder A]
    [Algebra 𝕜 A] : NonnegSpectrumClass 𝕜 A ↔ ∀ a : A, 0 ≤ a → ∀ x ∈ spectrum 𝕜 a, 0 ≤ x := by
  simp [show NonnegSpectrumClass 𝕜 A ↔ _ from ⟨fun ⟨h⟩ ↦ h, (⟨·⟩)⟩,
    quasispectrum_eq_spectrum_union_zero]


alias ⟨_, of_spectrum_nonneg⟩ := iff_spectrum_nonneg


lemma spectrum_nonneg_of_nonneg {𝕜 A : Type*} [OrderedCommSemiring 𝕜] [Ring A] [PartialOrder A]
    [Algebra 𝕜 A] [NonnegSpectrumClass 𝕜 A] ⦃a : A⦄ (ha : 0 ≤ a) ⦃x : 𝕜⦄ (hx : x ∈ spectrum 𝕜 a) :
    0 ≤ x :=
  NonnegSpectrumClass.quasispectrum_nonneg_of_nonneg a ha x (spectrum_subset_quasispectrum 𝕜 a hx)


/-- Given an element `a : A` of an `S`-algebra, where `S` is itself an `R`-algebra, we say that
the spectrum of `a` restricts via a function `f : S → R` if `f` is a left inverse of
`algebraMap R S`, and `f` is a right inverse of `algebraMap R S` on `spectrum S a`.

For example, when `f = Complex.re` (so `S := ℂ` and `R := ℝ`), `SpectrumRestricts a f` means that
the `ℂ`-spectrum of `a` is contained within `ℝ`. This arises naturally when `a` is selfadjoint
and `A` is a C⋆-algebra.

This is the property allows us to restrict a continuous functional calculus over `S` to a
continuous functional calculus over `R`. -/
structure QuasispectrumRestricts
    {R S A : Type*} [CommSemiring R] [CommSemiring S] [NonUnitalRing A]
    [Module R A] [Module S A] [Algebra R S] (a : A) (f : S → R) : Prop where
  /-- `f` is a right inverse of `algebraMap R S` when restricted to `quasispectrum S a`. -/
  rightInvOn : (quasispectrum S a).RightInvOn f (algebraMap R S)
  /-- `f` is a left inverse of `algebraMap R S`. -/
  left_inv : Function.LeftInverse f (algebraMap R S)


lemma quasispectrumRestricts_iff
    {R S A : Type*} [CommSemiring R] [CommSemiring S] [NonUnitalRing A]
    [Module R A] [Module S A] [Algebra R S] (a : A) (f : S → R) :
    QuasispectrumRestricts a f ↔ (quasispectrum S a).RightInvOn f (algebraMap R S) ∧
      Function.LeftInverse f (algebraMap R S) :=
  ⟨fun ⟨h₁, h₂⟩ ↦ ⟨h₁, h₂⟩, fun ⟨h₁, h₂⟩ ↦ ⟨h₁, h₂⟩⟩


@[simp]
theorem quasispectrum.algebraMap_mem_iff (S : Type*) {R A : Type*} [Semifield R] [Field S]
    [NonUnitalRing A] [Algebra R S] [Module S A] [IsScalarTower S A A]
    [SMulCommClass S A A] [Module R A] [IsScalarTower R S A] {a : A} {r : R} :
    algebraMap R S r ∈ quasispectrum S a ↔ r ∈ quasispectrum R a := by
  /-
    S : Type u_3
    R : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module S A
    inst✝³ : IsScalarTower S A A
    inst✝² : SMulCommClass S A A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R S A
    a : A
    r : R
    ⊢ Iff (Membership.mem (quasispectrum S a) ((algebraMap R S) r)) (Membership.me …
  -/
  simp_rw [Unitization.quasispectrum_eq_spectrum_inr' _ S a, spectrum.algebraMap_mem_iff]
  /-
    🎉 no goals
  -/


protected alias ⟨quasispectrum.of_algebraMap_mem, quasispectrum.algebraMap_mem⟩ :=
  quasispectrum.algebraMap_mem_iff


@[simp]
theorem quasispectrum.preimage_algebraMap (S : Type*) {R A : Type*} [Semifield R] [Field S]
    [NonUnitalRing A] [Algebra R S] [Module S A] [IsScalarTower S A A]
    [SMulCommClass S A A] [Module R A] [IsScalarTower R S A] {a : A} :
    algebraMap R S ⁻¹' quasispectrum S a = quasispectrum R a :=
  Set.ext fun _ => quasispectrum.algebraMap_mem_iff _


protected theorem map_zero (h : QuasispectrumRestricts a f) : f 0 = 0 := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁵ : Semifield R
    inst✝⁴ : Field S
    inst✝³ : NonUnitalRing A
    inst✝² : Module R A
    inst✝¹ : Module S A
    inst✝ : Algebra R S
    a : A
    f : S → R
    h : QuasispectrumRestricts a f
    ⊢ Eq (f 0) 0
  -/
  rw [← h.left_inv 0, map_zero (algebraMap R S)]
  /-
    🎉 no goals
  -/


theorem of_subset_range_algebraMap (hf : f.LeftInverse (algebraMap R S))
    (h : quasispectrum S a ⊆ Set.range (algebraMap R S)) : QuasispectrumRestricts a f where
                               /-
                                 R : Type u_3
                                 S : Type u_4
                                 A : Type u_5
                                 inst✝⁵ : Semifield R
                                 inst✝⁴ : Field S
                                 inst✝³ : NonUnitalRing A
                                 inst✝² : Module R A
                                 inst✝¹ : Module S A
                                 inst✝ : Algebra R S
                                 a : A
                                 f : S → R
                                 hf : Function.LeftInverse f ⇑(algebraMap R S)
                                 h : HasSubset.Subset (quasispectrum S a) (Set.range ⇑(algebraMap R S))
                                 s : S
                                 hs : Membership.mem (quasispectrum S a) s
                                 ⊢ Eq ((algebraMap R S) (f s)) s
                               -/
  rightInvOn := fun s hs => by obtain ⟨r, rfl⟩ := h hs; rw [hf r]
                                                        /-
                                                          🎉 no goals
                                                        -/
  left_inv := hf


lemma of_quasispectrum_eq {a b : A} {f : S → R} (ha : QuasispectrumRestricts a f)
    (h : quasispectrum S a = quasispectrum S b) : QuasispectrumRestricts b f where
  rightInvOn := h ▸ ha.rightInvOn
  left_inv := ha.left_inv


theorem algebraMap_image (h : QuasispectrumRestricts a f) :
    algebraMap R S '' quasispectrum R a = quasispectrum S a := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module R A
    inst✝⁴ : Module S A
    inst✝³ : Algebra R S
    a : A
    f : S → R
    inst✝² : IsScalarTower S A A
    inst✝¹ : SMulCommClass S A A
    inst✝ : IsScalarTower R S A
    h : QuasispectrumRestricts a f
    ⊢ Eq (Set.image (⇑(algebraMap R S)) (quasispectrum R a)) (quasispectrum S a)
  -/
  refine Set.eq_of_subset_of_subset ?_ fun s hs => ⟨f s, ?_⟩
  · simpa only [quasispectrum.preimage_algebraMap] using
      (quasispectrum S a).image_preimage_subset (algebraMap R S)
  /-
    case refine_2
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module R A
    inst✝⁴ : Module S A
    inst✝³ : Algebra R S
    a : A
    f : S → R
    inst✝² : IsScalarTower S A A
    inst✝¹ : SMulCommClass S A A
    inst✝ : IsScalarTower R S A
    h : QuasispectrumRestricts a f
    s : S
    hs : Membership.mem (quasispectrum S a) s
    ⊢ And (Membership.mem (quasispectrum R a) (f s)) (Eq ((algebraMap R S) (f s)) s)
  -/
  exact ⟨quasispectrum.of_algebraMap_mem S ((h.rightInvOn hs).symm ▸ hs), h.rightInvOn hs⟩
  /-
    🎉 no goals
  -/


theorem image (h : QuasispectrumRestricts a f) : f '' quasispectrum S a = quasispectrum R a := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁸ : Semifield R
    inst✝⁷ : Field S
    inst✝⁶ : NonUnitalRing A
    inst✝⁵ : Module R A
    inst✝⁴ : Module S A
    inst✝³ : Algebra R S
    a : A
    f : S → R
    inst✝² : IsScalarTower S A A
    inst✝¹ : SMulCommClass S A A
    inst✝ : IsScalarTower R S A
    h : QuasispectrumRestricts a f
    ⊢ Eq (Set.image f (quasispectrum S a)) (quasispectrum R a)
  -/
  simp only [← h.algebraMap_image, Set.image_image, h.left_inv _, Set.image_id']
  /-
    🎉 no goals
  -/


theorem apply_mem (h : QuasispectrumRestricts a f) {s : S} (hs : s ∈ quasispectrum S a) :
    f s ∈ quasispectrum R a :=
  h.image ▸ ⟨s, hs, rfl⟩


theorem subset_preimage (h : QuasispectrumRestricts a f) :
    quasispectrum S a ⊆ f ⁻¹' quasispectrum R a :=
  h.image ▸ (quasispectrum S a).subset_preimage_image f


protected lemma comp {R₁ R₂ R₃ A : Type*} [Semifield R₁] [Field R₂] [Field R₃]
    [NonUnitalRing A] [Module R₁ A] [Module R₂ A] [Module R₃ A] [Algebra R₁ R₂] [Algebra R₂ R₃]
    [Algebra R₁ R₃] [IsScalarTower R₁ R₂ R₃] [IsScalarTower R₂ R₃ A] [IsScalarTower R₃ A A]
    [SMulCommClass R₃ A A] {a : A} {f : R₃ → R₂} {g : R₂ → R₁} {e : R₃ → R₁} (hfge : g ∘ f = e)
    (hf : QuasispectrumRestricts a f) (hg : QuasispectrumRestricts a g) :
    QuasispectrumRestricts a e where
  left_inv := by
    /-
      R₁ : Type u_6
      R₂ : Type u_7
      R₃ : Type u_8
      A : Type u_9
      inst✝¹³ : Semifield R₁
      inst✝¹² : Field R₂
      inst✝¹¹ : Field R₃
      inst✝¹⁰ : NonUnitalRing A
      inst✝⁹ : Module R₁ A
      inst✝⁸ : Module R₂ A
      inst✝⁷ : Module R₃ A
      inst✝⁶ : Algebra R₁ R₂
      inst✝⁵ : Algebra R₂ R₃
      inst✝⁴ : Algebra R₁ R₃
      inst✝³ : IsScalarTower R₁ R₂ R₃
      inst✝² : IsScalarTower R₂ R₃ A
      inst✝¹ : IsScalarTower R₃ A A
      inst✝ : SMulCommClass R₃ A A
      a : A
      f : R₃ → R₂
      g : R₂ → R₁
      e : R₃ → R₁
      hfge : Eq (Function.comp g f) e
      hf : QuasispectrumRestricts a f
      hg : QuasispectrumRestricts a g
      ⊢ Function.LeftInverse e ⇑(algebraMap R₁ R₃)
    -/
    convert hfge ▸ hf.left_inv.comp hg.left_inv
    /-
      case h.e'_4
      R₁ : Type u_6
      R₂ : Type u_7
      R₃ : Type u_8
      A : Type u_9
      inst✝¹³ : Semifield R₁
      inst✝¹² : Field R₂
      inst✝¹¹ : Field R₃
      inst✝¹⁰ : NonUnitalRing A
      inst✝⁹ : Module R₁ A
      inst✝⁸ : Module R₂ A
      inst✝⁷ : Module R₃ A
      inst✝⁶ : Algebra R₁ R₂
      inst✝⁵ : Algebra R₂ R₃
      inst✝⁴ : Algebra R₁ R₃
      inst✝³ : IsScalarTower R₁ R₂ R₃
      inst✝² : IsScalarTower R₂ R₃ A
      inst✝¹ : IsScalarTower R₃ A A
      inst✝ : SMulCommClass R₃ A A
      a : A
      f : R₃ → R₂
      g : R₂ → R₁
      e : R₃ → R₁
      hfge : Eq (Function.comp g f) e
      hf : QuasispectrumRestricts a f
      hg : QuasispectrumRestricts a g
      ⊢ Eq (⇑(algebraMap R₁ R₃)) (Function.comp ⇑(algebraMap R₂ R₃) ⇑(algebraMap R₁  …
    -/
    /-
      R₁ : Type u_6
      R₂ : Type u_7
      R₃ : Type u_8
      A : Type u_9
      inst✝¹³ : Semifield R₁
      inst✝¹² : Field R₂
      inst✝¹¹ : Field R₃
      inst✝¹⁰ : NonUnitalRing A
      inst✝⁹ : Module R₁ A
      inst✝⁸ : Module R₂ A
      inst✝⁷ : Module R₃ A
      inst✝⁶ : Algebra R₁ R₂
      inst✝⁵ : Algebra R₂ R₃
      inst✝⁴ : Algebra R₁ R₃
      inst✝³ : IsScalarTower R₁ R₂ R₃
      inst✝² : IsScalarTower R₂ R₃ A
      inst✝¹ : IsScalarTower R₃ A A
      inst✝ : SMulCommClass R₃ A A
      a : A
      f : R₃ → R₂
      g : R₂ → R₁
      e : R₃ → R₁
      hfge : Eq (Function.comp g f) e
      hf : QuasispectrumRestricts a f
      hg : QuasispectrumRestricts a g
      ⊢ Set.RightInvOn e (⇑(algebraMap R₁ R₃)) (quasispectrum R₃ a)
    -/
    congrm(⇑$(IsScalarTower.algebraMap_eq R₁ R₂ R₃))
    /-
      case h.e'_4
      R₁ : Type u_6
      R₂ : Type u_7
      R₃ : Type u_8
      A : Type u_9
      inst✝¹³ : Semifield R₁
      inst✝¹² : Field R₂
      inst✝¹¹ : Field R₃
      inst✝¹⁰ : NonUnitalRing A
      inst✝⁹ : Module R₁ A
      inst✝⁸ : Module R₂ A
      inst✝⁷ : Module R₃ A
      inst✝⁶ : Algebra R₁ R₂
      inst✝⁵ : Algebra R₂ R₃
      inst✝⁴ : Algebra R₁ R₃
      inst✝³ : IsScalarTower R₁ R₂ R₃
      inst✝² : IsScalarTower R₂ R₃ A
      inst✝¹ : IsScalarTower R₃ A A
      inst✝ : SMulCommClass R₃ A A
      a : A
      f : R₃ → R₂
      g : R₂ → R₁
      e : R₃ → R₁
      hfge : Eq (Function.comp g f) e
      hf : QuasispectrumRestricts a f
      hg : QuasispectrumRestricts a g
      ⊢ Eq (⇑(algebraMap R₁ R₃)) (Function.comp ⇑(algebraMap R₂ R₃) ⇑(algebraMap R₁  …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  rightInvOn := by
    convert hfge ▸ hg.rightInvOn.comp hf.rightInvOn fun _ ↦ hf.apply_mem
    congrm(⇑$(IsScalarTower.algebraMap_eq R₁ R₂ R₃))


/-- A (reducible) alias of `QuasispectrumRestricts` which enforces stronger type class assumptions
on the types involved, as it's really intended for the `spectrum`. The separate definition also
allows for dot notation. -/
@[reducible]
def SpectrumRestricts
    {R S A : Type*} [Semifield R] [Semifield S] [Ring A]
    [Algebra R A] [Algebra S A] [Algebra R S] (a : A) (f : S → R) : Prop :=
  QuasispectrumRestricts a f


theorem rightInvOn (h : SpectrumRestricts a f) : (spectrum S a).RightInvOn f (algebraMap R S) :=
  (QuasispectrumRestricts.rightInvOn h).mono <| spectrum_subset_quasispectrum _ _


theorem of_rightInvOn (h₁ : Function.LeftInverse f (algebraMap R S))
    (h₂ : (spectrum S a).RightInvOn f (algebraMap R S)) : SpectrumRestricts a f where
  rightInvOn x hx := by
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a : A
      f : S → R
      h₁ : Function.LeftInverse f ⇑(algebraMap R S)
      h₂ : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
      x : S
      hx : Membership.mem (quasispectrum S a) x
      ⊢ Eq ((algebraMap R S) (f x)) x
    -/
    obtain (rfl | hx) := mem_quasispectrum_iff.mp hx
      /-
        case inl
        R : Type u_3
        S : Type u_4
        A : Type u_5
        inst✝⁵ : Semifield R
        inst✝⁴ : Semifield S
        inst✝³ : Ring A
        inst✝² : Algebra R S
        inst✝¹ : Algebra R A
        inst✝ : Algebra S A
        a : A
        f : S → R
        h₁ : Function.LeftInverse f ⇑(algebraMap R S)
        h₂ : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
        hx : Membership.mem (quasispectrum S a) 0
        ⊢ Eq ((algebraMap R S) (f 0)) 0
      -/
    · simpa using h₁ 0
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u_3
        S : Type u_4
        A : Type u_5
        inst✝⁵ : Semifield R
        inst✝⁴ : Semifield S
        inst✝³ : Ring A
        inst✝² : Algebra R S
        inst✝¹ : Algebra R A
        inst✝ : Algebra S A
        a : A
        f : S → R
        h₁ : Function.LeftInverse f ⇑(algebraMap R S)
        h₂ : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
        x : S
        hx✝ : Membership.mem (quasispectrum S a) x
        hx : Membership.mem (spectrum S a) x
        ⊢ Eq ((algebraMap R S) (f x)) x
      -/
    · exact h₂ hx
      /-
        🎉 no goals
      -/
  left_inv := h₁


lemma _root_.spectrumRestricts_iff :
    SpectrumRestricts a f ↔ (spectrum S a).RightInvOn f (algebraMap R S) ∧
      Function.LeftInverse f (algebraMap R S) :=
  ⟨fun h ↦ ⟨h.rightInvOn, h.left_inv⟩, fun h ↦ .of_rightInvOn h.2 h.1⟩


theorem of_subset_range_algebraMap (hf : f.LeftInverse (algebraMap R S))
    (h : spectrum S a ⊆ Set.range (algebraMap R S)) : SpectrumRestricts a f where
  rightInvOn := fun s hs => by
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a : A
      f : S → R
      hf : Function.LeftInverse f ⇑(algebraMap R S)
      h : HasSubset.Subset (spectrum S a) (Set.range ⇑(algebraMap R S))
      s : S
      hs : Membership.mem (quasispectrum S a) s
      ⊢ Eq ((algebraMap R S) (f s)) s
    -/
    rw [mem_quasispectrum_iff] at hs
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a : A
      f : S → R
      hf : Function.LeftInverse f ⇑(algebraMap R S)
      h : HasSubset.Subset (spectrum S a) (Set.range ⇑(algebraMap R S))
      s : S
      hs : Or (Eq s 0) (Membership.mem (spectrum S a) s)
      ⊢ Eq ((algebraMap R S) (f s)) s
    -/
    obtain (rfl | hs) := hs
      /-
        case inl
        R : Type u_3
        S : Type u_4
        A : Type u_5
        inst✝⁵ : Semifield R
        inst✝⁴ : Semifield S
        inst✝³ : Ring A
        inst✝² : Algebra R S
        inst✝¹ : Algebra R A
        inst✝ : Algebra S A
        a : A
        f : S → R
        hf : Function.LeftInverse f ⇑(algebraMap R S)
        h : HasSubset.Subset (spectrum S a) (Set.range ⇑(algebraMap R S))
        ⊢ Eq ((algebraMap R S) (f 0)) 0
      -/
    · simpa using hf 0
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u_3
        S : Type u_4
        A : Type u_5
        inst✝⁵ : Semifield R
        inst✝⁴ : Semifield S
        inst✝³ : Ring A
        inst✝² : Algebra R S
        inst✝¹ : Algebra R A
        inst✝ : Algebra S A
        a : A
        f : S → R
        hf : Function.LeftInverse f ⇑(algebraMap R S)
        h : HasSubset.Subset (spectrum S a) (Set.range ⇑(algebraMap R S))
        s : S
        hs : Membership.mem (spectrum S a) s
        ⊢ Eq ((algebraMap R S) (f s)) s
      -/
    · obtain ⟨r, rfl⟩ := h hs
      /-
        case inr.intro
        R : Type u_3
        S : Type u_4
        A : Type u_5
        inst✝⁵ : Semifield R
        inst✝⁴ : Semifield S
        inst✝³ : Ring A
        inst✝² : Algebra R S
        inst✝¹ : Algebra R A
        inst✝ : Algebra S A
        a : A
        f : S → R
        hf : Function.LeftInverse f ⇑(algebraMap R S)
        h : HasSubset.Subset (spectrum S a) (Set.range ⇑(algebraMap R S))
        r : R
        hs : Membership.mem (spectrum S a) ((algebraMap R S) r)
        ⊢ Eq ((algebraMap R S) (f ((algebraMap R S) r))) ((algebraMap R S) r)
      -/
      rw [hf r]
      /-
        🎉 no goals
      -/
  left_inv := hf


lemma of_spectrum_eq {a b : A} {f : S → R} (ha : SpectrumRestricts a f)
    (h : spectrum S a = spectrum S b) : SpectrumRestricts b f where
  rightInvOn :=  by
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a b : A
      f : S → R
      ha : SpectrumRestricts a f
      h : Eq (spectrum S a) (spectrum S b)
      ⊢ Set.RightInvOn f (⇑(algebraMap R S)) (quasispectrum S b)
    -/
    rw [quasispectrum_eq_spectrum_union_zero, ← h, ← quasispectrum_eq_spectrum_union_zero]
    /-
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a b : A
      f : S → R
      ha : SpectrumRestricts a f
      h : Eq (spectrum S a) (spectrum S b)
      ⊢ Set.RightInvOn f (⇑(algebraMap R S)) (quasispectrum S a)
    -/
    exact QuasispectrumRestricts.rightInvOn ha
    /-
      🎉 no goals
    -/
  left_inv := ha.left_inv


theorem algebraMap_image (h : SpectrumRestricts a f) :
    algebraMap R S '' spectrum R a = spectrum S a := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁶ : Semifield R
    inst✝⁵ : Semifield S
    inst✝⁴ : Ring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    a : A
    f : S → R
    inst✝ : IsScalarTower R S A
    h : SpectrumRestricts a f
    ⊢ Eq (Set.image (⇑(algebraMap R S)) (spectrum R a)) (spectrum S a)
  -/
  refine Set.eq_of_subset_of_subset ?_ fun s hs => ⟨f s, ?_⟩
  · simpa only [spectrum.preimage_algebraMap] using
      (spectrum S a).image_preimage_subset (algebraMap R S)
  /-
    case refine_2
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁶ : Semifield R
    inst✝⁵ : Semifield S
    inst✝⁴ : Ring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    a : A
    f : S → R
    inst✝ : IsScalarTower R S A
    h : SpectrumRestricts a f
    s : S
    hs : Membership.mem (spectrum S a) s
    ⊢ And (Membership.mem (spectrum R a) (f s)) (Eq ((algebraMap R S) (f s)) s)
  -/
  exact ⟨spectrum.of_algebraMap_mem S ((h.rightInvOn hs).symm ▸ hs), h.rightInvOn hs⟩
  /-
    🎉 no goals
  -/


theorem image (h : SpectrumRestricts a f) : f '' spectrum S a = spectrum R a := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁶ : Semifield R
    inst✝⁵ : Semifield S
    inst✝⁴ : Ring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    a : A
    f : S → R
    inst✝ : IsScalarTower R S A
    h : SpectrumRestricts a f
    ⊢ Eq (Set.image f (spectrum S a)) (spectrum R a)
  -/
  simp only [← h.algebraMap_image, Set.image_image, h.left_inv _, Set.image_id']
  /-
    🎉 no goals
  -/


theorem apply_mem (h : SpectrumRestricts a f) {s : S} (hs : s ∈ spectrum S a) :
    f s ∈ spectrum R a :=
  h.image ▸ ⟨s, hs, rfl⟩


theorem subset_preimage (h : SpectrumRestricts a f) : spectrum S a ⊆ f ⁻¹' spectrum R a :=
  h.image ▸ (spectrum S a).subset_preimage_image f


theorem quasispectrumRestricts_iff_spectrumRestricts_inr (S : Type*) {R A : Type*} [Semifield R]
    [Field S] [NonUnitalRing A] [Algebra R S] [Module R A] [Module S A] [IsScalarTower S A A]
    [SMulCommClass S A A] [IsScalarTower R S A] {a : A} {f : S → R} :
    QuasispectrumRestricts a f ↔ SpectrumRestricts (a : Unitization S A) f := by
  rw [quasispectrumRestricts_iff, spectrumRestricts_iff,
    ← Unitization.quasispectrum_eq_spectrum_inr']


theorem quasispectrumRestricts_iff_spectrumRestricts {R S A : Type*} [Semifield R] [Semifield S]
    [Ring A] [Algebra R S] [Algebra R A] [Algebra S A] {a : A} {f : S → R} :
    QuasispectrumRestricts a f ↔ SpectrumRestricts a f := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁵ : Semifield R
    inst✝⁴ : Semifield S
    inst✝³ : Ring A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra S A
    a : A
    f : S → R
    ⊢ Iff (QuasispectrumRestricts a f) (SpectrumRestricts a f)
  -/
  rw [quasispectrumRestricts_iff, spectrumRestricts_iff, quasispectrum_eq_spectrum_union_zero]
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁵ : Semifield R
    inst✝⁴ : Semifield S
    inst✝³ : Ring A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra S A
    a : A
    f : S → R
    ⊢ Iff (And (Set.RightInvOn f (⇑(algebraMap R S)) (Union.union (spectrum S a) ( …
  -/
  refine and_congr_left fun h ↦ ?_
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁵ : Semifield R
    inst✝⁴ : Semifield S
    inst✝³ : Ring A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra S A
    a : A
    f : S → R
    h : Function.LeftInverse f ⇑(algebraMap R S)
    ⊢ Iff (Set.RightInvOn f (⇑(algebraMap R S)) (Union.union (spectrum S a) (Singl …
  -/
  refine ⟨(Set.RightInvOn.mono · Set.subset_union_left), fun h' x hx ↦ ?_⟩
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁵ : Semifield R
    inst✝⁴ : Semifield S
    inst✝³ : Ring A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra S A
    a : A
    f : S → R
    h : Function.LeftInverse f ⇑(algebraMap R S)
    h' : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
    x : S
    hx : Membership.mem (Union.union (spectrum S a) (Singleton.singleton 0)) x
    ⊢ Eq ((algebraMap R S) (f x)) x
  -/
  simp only [Set.union_singleton, Set.mem_insert_iff] at hx
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝⁵ : Semifield R
    inst✝⁴ : Semifield S
    inst✝³ : Ring A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra S A
    a : A
    f : S → R
    h : Function.LeftInverse f ⇑(algebraMap R S)
    h' : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
    x : S
    hx : Or (Eq x 0) (Membership.mem (spectrum S a) x)
    ⊢ Eq ((algebraMap R S) (f x)) x
  -/
  obtain (rfl | hx) := hx
    /-
      case inl
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a : A
      f : S → R
      h : Function.LeftInverse f ⇑(algebraMap R S)
      h' : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
      ⊢ Eq ((algebraMap R S) (f 0)) 0
    -/
  · simpa using h 0
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁵ : Semifield R
      inst✝⁴ : Semifield S
      inst✝³ : Ring A
      inst✝² : Algebra R S
      inst✝¹ : Algebra R A
      inst✝ : Algebra S A
      a : A
      f : S → R
      h : Function.LeftInverse f ⇑(algebraMap R S)
      h' : Set.RightInvOn f (⇑(algebraMap R S)) (spectrum S a)
      x : S
      hx : Membership.mem (spectrum S a) x
      ⊢ Eq ((algebraMap R S) (f x)) x
    -/
  · exact h' hx
    /-
      🎉 no goals
    -/

