/-- A `EuclideanDomain` is a non-trivial commutative ring with a division and a remainder,
  satisfying `b * (a / b) + a % b = a`.
  The definition of a Euclidean domain usually includes a valuation function `R → ℕ`.
  This definition is slightly generalised to include a well founded relation
  `r` with the property that `r (a % b) b`, instead of a valuation. -/
class EuclideanDomain (R : Type u) extends CommRing R, Nontrivial R where
  /-- A division function (denoted `/`) on `R`.
    This satisfies the property `b * (a / b) + a % b = a`, where `%` denotes `remainder`. -/
  protected quotient : R → R → R
  /-- Division by zero should always give zero by convention. -/
  protected quotient_zero : ∀ a, quotient a 0 = 0
  /-- A remainder function (denoted `%`) on `R`.
    This satisfies the property `b * (a / b) + a % b = a`, where `/` denotes `quotient`. -/
  protected remainder : R → R → R
  /-- The property that links the quotient and remainder functions.
    This allows us to compute GCDs and LCMs. -/
  protected quotient_mul_add_remainder_eq : ∀ a b, b * quotient a b + remainder a b = a
  /-- A well-founded relation on `R`, satisfying `r (a % b) b`.
    This ensures that the GCD algorithm always terminates. -/
  protected r : R → R → Prop
  /-- The relation `r` must be well-founded.
    This ensures that the GCD algorithm always terminates. -/
  r_wellFounded : WellFounded r
  /-- The relation `r` satisfies `r (a % b) b`. -/
  protected remainder_lt : ∀ (a) {b}, b ≠ 0 → r (remainder a b) b
  /-- An additional constraint on `r`. -/
  mul_left_not_lt : ∀ (a) {b}, b ≠ 0 → ¬r (a * b) a


/-- Abbreviated notation for the well-founded relation `r` in a Euclidean domain. -/
local infixl:50 " ≺ " => EuclideanDomain.r


local instance wellFoundedRelation : WellFoundedRelation R where
  wf := r_wellFounded


instance isWellFounded : IsWellFounded R (· ≺ ·) where
  wf := r_wellFounded

-- see Note [lower instance priority]

instance (priority := 70) : Div R :=
  ⟨EuclideanDomain.quotient⟩

-- see Note [lower instance priority]

instance (priority := 70) : Mod R :=
  ⟨EuclideanDomain.remainder⟩


theorem div_add_mod (a b : R) : b * (a / b) + a % b = a :=
  EuclideanDomain.quotient_mul_add_remainder_eq _ _


theorem mod_add_div (a b : R) : a % b + b * (a / b) = a :=
  (add_comm _ _).trans (div_add_mod _ _)


theorem mod_add_div' (m k : R) : m % k + m / k * k = m := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    m k : R
    ⊢ Eq (HAdd.hAdd (HMod.hMod m k) (HMul.hMul (HDiv.hDiv m k) k)) m
  -/
  rw [mul_comm]
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    m k : R
    ⊢ Eq (HAdd.hAdd (HMod.hMod m k) (HMul.hMul k (HDiv.hDiv m k))) m
  -/
  exact mod_add_div _ _
  /-
    🎉 no goals
  -/


theorem div_add_mod' (m k : R) : m / k * k + m % k = m := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    m k : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv m k) k) (HMod.hMod m k)) m
  -/
  rw [mul_comm]
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    m k : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul k (HDiv.hDiv m k)) (HMod.hMod m k)) m
  -/
  exact div_add_mod _ _
  /-
    🎉 no goals
  -/


theorem mod_eq_sub_mul_div {R : Type*} [EuclideanDomain R] (a b : R) : a % b = a - b * (a / b) :=
  calc
    a % b = b * (a / b) + a % b - b * (a / b) := (add_sub_cancel_left _ _).symm
                              /-
                                R : Type u_1
                                inst✝ : EuclideanDomain R
                                a b : R
                                ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul b (HDiv.hDiv a b)) (HMod.hMod a b)) (HMu …
                              -/
    _ = a - b * (a / b) := by rw [div_add_mod]
                              /-
                                🎉 no goals
                              -/


theorem mod_lt : ∀ (a) {b : R}, b ≠ 0 → a % b ≺ b :=
  EuclideanDomain.remainder_lt


theorem mul_right_not_lt {a : R} (b) (h : a ≠ 0) : ¬a * b ≺ b := by
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b : R
    h : Ne a 0
    ⊢ Not (EuclideanDomain.r (HMul.hMul a b) b)
  -/
  rw [mul_comm]
  /-
    R : Type u
    inst✝ : EuclideanDomain R
    a b : R
    h : Ne a 0
    ⊢ Not (EuclideanDomain.r (HMul.hMul b a) b)
  -/
  exact mul_left_not_lt b h
  /-
    🎉 no goals
  -/


@[simp]
                                           /-
                                             R : Type u
                                             inst✝ : EuclideanDomain R
                                             a : R
                                             ⊢ Eq (HMod.hMod a 0) a
                                           -/
theorem mod_zero (a : R) : a % 0 = a := by simpa only [zero_mul, zero_add] using div_add_mod a 0
                                           /-
                                             🎉 no goals
                                           -/


theorem lt_one (a : R) : a ≺ (1 : R) → a = 0 :=
  haveI := Classical.dec
                            /-
                              R : Type u
                              inst✝ : EuclideanDomain R
                              a : R
                              this : (p : Prop) → Decidable p
                              h : Not (Eq a 0)
                              ⊢ Not (EuclideanDomain.r a 1)
                            -/
  not_imp_not.1 fun h => by simpa only [one_mul] using mul_left_not_lt 1 h
                            /-
                              🎉 no goals
                            -/


theorem val_dvd_le : ∀ a b : R, b ∣ a → a ≠ 0 → ¬a ≺ b
                                                    /-
                                                      R : Type u
                                                      inst✝ : EuclideanDomain R
                                                      b d : R
                                                      ha : Ne (HMul.hMul b d) 0
                                                      ⊢ Eq d 0 → Eq (HMul.hMul b d) 0
                                                    -/
  | _, b, ⟨d, rfl⟩, ha => mul_left_not_lt b (mt (by rintro rfl; exact mul_zero _) ha)
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem div_zero (a : R) : a / 0 = 0 :=
  EuclideanDomain.quotient_zero a


@[elab_as_elim]
theorem GCD.induction {P : R → R → Prop} (a b : R) (H0 : ∀ x, P 0 x)
    (H1 : ∀ a b, a ≠ 0 → P (b % a) a → P a b) : P a b := by
  classical
  exact if a0 : a = 0 then
    a0.symm ▸ H0 b
  else
    have _ := mod_lt b a0
    H1 _ _ a0 (GCD.induction (b % a) a H0 H1)
termination_by a


/-- `gcd a b` is a (non-unique) element such that `gcd a b ∣ a` `gcd a b ∣ b`, and for
  any element `c` such that `c ∣ a` and `c ∣ b`, then `c ∣ gcd a b` -/
def gcd (a b : R) : R :=
  if a0 : a = 0 then b
  else
    have _ := mod_lt b a0
    gcd (b % a) a
termination_by a


@[simp]
theorem gcd_zero_left (a : R) : gcd 0 a = a := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a : R
    ⊢ Eq (EuclideanDomain.gcd 0 a) a
  -/
  rw [gcd]
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    a : R
    ⊢ Eq (dite (Eq 0 0) (fun a0 => a) fun a0 => letFun ⋯ fun x => EuclideanDomain. …
  -/
  exact if_pos rfl
  /-
    🎉 no goals
  -/


/-- An implementation of the extended GCD algorithm.
At each step we are computing a triple `(r, s, t)`, where `r` is the next value of the GCD
algorithm, to compute the greatest common divisor of the input (say `x` and `y`), and `s` and `t`
are the coefficients in front of `x` and `y` to obtain `r` (i.e. `r = s * x + t * y`).
The function `xgcdAux` takes in two triples, and from these recursively computes the next triple:
```
xgcdAux (r, s, t) (r', s', t') = xgcdAux (r' % r, s' - (r' / r) * s, t' - (r' / r) * t) (r, s, t)
```
-/
def xgcdAux (r s t r' s' t' : R) : R × R × R :=
  if _hr : r = 0 then (r', s', t')
  else
    let q := r' / r
    have _ := mod_lt r' _hr
    xgcdAux (r' % r) (s' - q * s) (t' - q * t) r s t
termination_by r


@[simp]
theorem xgcd_zero_left {s t r' s' t' : R} : xgcdAux 0 s t r' s' t' = (r', s', t') := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    s t r' s' t' : R
    ⊢ Eq (EuclideanDomain.xgcdAux 0 s t r' s' t') { fst := r', snd := { fst := s', …
  -/
  unfold xgcdAux
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    s t r' s' t' : R
    ⊢ Eq
        (dite (Eq 0 0) (fun _hr => { fst := r', snd := { fst := s', snd := t' } }) …
          let q := HDiv.hDiv r' 0;
          letFun ⋯ fun x => EuclideanDomain.xgcdAux (HMod.hMod r' 0) (HSub.hSub s' …
        { fst := r', snd := { fst := s', snd := t' } }
  -/
  exact if_pos rfl
  /-
    🎉 no goals
  -/


theorem xgcdAux_rec {r s t r' s' t' : R} (h : r ≠ 0) :
    xgcdAux r s t r' s' t' = xgcdAux (r' % r) (s' - r' / r * s) (t' - r' / r * t) r s t := by
  conv =>
    lhs
    rw [xgcdAux]
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    r s t r' s' t' : R
    h : Ne r 0
    ⊢ Eq
        (dite (Eq r 0) (fun _hr => { fst := r', snd := { fst := s', snd := t' } }) …
          let q := HDiv.hDiv r' r;
          letFun ⋯ fun x => EuclideanDomain.xgcdAux (HMod.hMod r' r) (HSub.hSub s' …
        (EuclideanDomain.xgcdAux (HMod.hMod r' r) (HSub.hSub s' (HMul.hMul (HDiv.h …
  -/
  exact if_neg h
  /-
    🎉 no goals
  -/


/-- Use the extended GCD algorithm to generate the `a` and `b` values
  satisfying `gcd x y = x * a + y * b`. -/
def xgcd (x y : R) : R × R :=
  (xgcdAux x 1 0 y 0 1).2


/-- The extended GCD `a` value in the equation `gcd x y = x * a + y * b`. -/
def gcdA (x y : R) : R :=
  (xgcd x y).1


/-- The extended GCD `b` value in the equation `gcd x y = x * a + y * b`. -/
def gcdB (x y : R) : R :=
  (xgcd x y).2


@[simp]
theorem gcdA_zero_left {s : R} : gcdA 0 s = 0 := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    s : R
    ⊢ Eq (EuclideanDomain.gcdA 0 s) 0
  -/
  unfold gcdA
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    s : R
    ⊢ Eq (EuclideanDomain.xgcd 0 s).fst 0
  -/
  rw [xgcd, xgcd_zero_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcdB_zero_left {s : R} : gcdB 0 s = 1 := by
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    s : R
    ⊢ Eq (EuclideanDomain.gcdB 0 s) 1
  -/
  unfold gcdB
  /-
    R : Type u
    inst✝¹ : EuclideanDomain R
    inst✝ : DecidableEq R
    s : R
    ⊢ Eq (EuclideanDomain.xgcd 0 s).snd 1
  -/
  rw [xgcd, xgcd_zero_left]
  /-
    🎉 no goals
  -/


theorem xgcd_val (x y : R) : xgcd x y = (gcdA x y, gcdB x y) :=
  rfl


/-- `lcm a b` is a (non-unique) element such that `a ∣ lcm a b` `b ∣ lcm a b`, and for
  any element `c` such that `a ∣ c` and `b ∣ c`, then `lcm a b ∣ c` -/
def lcm (x y : R) : R :=
  x * y / gcd x y


