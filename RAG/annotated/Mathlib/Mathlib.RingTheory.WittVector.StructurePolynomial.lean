set_option quotPrecheck false in
@[inherit_doc]
scoped[Witt] notation "W_" => wittPolynomial p

-- Notation with ring of coefficients implicit

set_option quotPrecheck false in
@[inherit_doc]
scoped[Witt] notation "W" => wittPolynomial p _


/-- `wittStructureRat Φ` is a family of polynomials `ℕ → MvPolynomial (idx × ℕ) ℚ`
that are uniquely characterised by the property that
```
bind₁ (wittStructureRat p Φ) (wittPolynomial p ℚ n) =
bind₁ (fun i ↦ (rename (prod.mk i) (wittPolynomial p ℚ n))) Φ
```
In other words: evaluating the `n`-th Witt polynomial on the family `wittStructureRat Φ`
is the same as evaluating `Φ` on the (appropriately renamed) `n`-th Witt polynomials.

See `wittStructureRat_prop` for this property,
and `wittStructureRat_existsUnique` for the fact that `wittStructureRat`
gives the unique family of polynomials with this property.

These polynomials turn out to have integral coefficients,
but it requires some effort to show this.
See `wittStructureInt` for the version with integral coefficients,
and `map_wittStructureInt` for the fact that it is equal to `wittStructureRat`
when mapped to polynomials over the rationals. -/
noncomputable def wittStructureRat (Φ : MvPolynomial idx ℚ) (n : ℕ) : MvPolynomial (idx × ℕ) ℚ :=
  bind₁ (fun k => bind₁ (fun i => rename (Prod.mk i) (W_ ℚ k)) Φ) (xInTermsOfW p ℚ n)


theorem wittStructureRat_prop (Φ : MvPolynomial idx ℚ) (n : ℕ) :
    bind₁ (wittStructureRat p Φ) (W_ ℚ n) = bind₁ (fun i => rename (Prod.mk i) (W_ ℚ n)) Φ :=
  calc
    bind₁ (wittStructureRat p Φ) (W_ ℚ n) =
        bind₁ (fun k => bind₁ (fun i => (rename (Prod.mk i)) (W_ ℚ k)) Φ)
          (bind₁ (xInTermsOfW p ℚ) (W_ ℚ n)) := by
      /-
        p : Nat
        idx : Type u_2
        hp : Fact (Nat.Prime p)
        Φ : MvPolynomial idx Rat
        n : Nat
        ⊢ Eq ((MvPolynomial.bind₁ (wittStructureRat p Φ)) (wittPolynomial p Rat n)) (( …
      -/
      rw [bind₁_bind₁]; exact eval₂Hom_congr (RingHom.ext_rat _ _) rfl rfl
                        /-
                          🎉 no goals
                        -/
    _ = bind₁ (fun i => rename (Prod.mk i) (W_ ℚ n)) Φ := by
      /-
        p : Nat
        idx : Type u_2
        hp : Fact (Nat.Prime p)
        Φ : MvPolynomial idx Rat
        n : Nat
        ⊢ Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => (MvPolynomial. …
      -/
      rw [bind₁_xInTermsOfW_wittPolynomial p _ n, bind₁_X_right]
      /-
        🎉 no goals
      -/


theorem wittStructureRat_existsUnique (Φ : MvPolynomial idx ℚ) :
    ∃! φ : ℕ → MvPolynomial (idx × ℕ) ℚ,
      ∀ n : ℕ, bind₁ φ (W_ ℚ n) = bind₁ (fun i => rename (Prod.mk i) (W_ ℚ n)) Φ := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    ⊢ ExistsUnique fun φ => ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomia …
  -/
  refine ⟨wittStructureRat p Φ, ?_, ?_⟩
    /-
      case refine_1
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Rat
      ⊢ (fun φ => ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Rat n))  …
    -/
  · intro n; apply wittStructureRat_prop
             /-
               🎉 no goals
             -/
    /-
      case refine_2
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Rat
      ⊢ ∀ (y : Nat → MvPolynomial (Prod idx Nat) Rat), (fun φ => ∀ (n : Nat), Eq ((M …
    -/
  · intro φ H
    /-
      case refine_2
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Rat
      φ : Nat → MvPolynomial (Prod idx Nat) Rat
      H : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Rat n)) ((MvPoly …
      ⊢ Eq φ (wittStructureRat p Φ)
    -/
    funext n
    rw [show φ n = bind₁ φ (bind₁ (W_ ℚ) (xInTermsOfW p ℚ n)) by
        rw [bind₁_wittPolynomial_xInTermsOfW p, bind₁_X_right]]
    /-
      case refine_2.h
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Rat
      φ : Nat → MvPolynomial (Prod idx Nat) Rat
      H : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Rat n)) ((MvPoly …
      n : Nat
      ⊢ Eq ((MvPolynomial.bind₁ φ) ((MvPolynomial.bind₁ (wittPolynomial p Rat)) (xIn …
    -/
    rw [bind₁_bind₁]
    /-
      case refine_2.h
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Rat
      φ : Nat → MvPolynomial (Prod idx Nat) Rat
      H : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Rat n)) ((MvPoly …
      n : Nat
      ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.bind₁ φ) (wittPolynomial p Ra …
    -/
    exact eval₂Hom_congr (RingHom.ext_rat _ _) (funext H) rfl
    /-
      🎉 no goals
    -/


theorem wittStructureRat_rec_aux (Φ : MvPolynomial idx ℚ) (n : ℕ) :
    wittStructureRat p Φ n * C ((p : ℚ) ^ n) =
      bind₁ (fun b => rename (fun i => (b, i)) (W_ ℚ n)) Φ -
        ∑ i ∈ range n, C ((p : ℚ) ^ i) * wittStructureRat p Φ i ^ p ^ (n - i) := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Eq (HMul.hMul (wittStructureRat p Φ n) (MvPolynomial.C (HPow.hPow (↑p) n)))  …
  -/
  have := xInTermsOfW_aux p ℚ n
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    this : Eq (HMul.hMul (xInTermsOfW p Rat n) (MvPolynomial.C (HPow.hPow (↑p) n)) …
    ⊢ Eq (HMul.hMul (wittStructureRat p Φ n) (MvPolynomial.C (HPow.hPow (↑p) n)))  …
  -/
  replace := congr_arg (bind₁ fun k : ℕ => bind₁ (fun i => rename (Prod.mk i) (W_ ℚ k)) Φ) this
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    this : Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => (MvPolyno …
    ⊢ Eq (HMul.hMul (wittStructureRat p Φ n) (MvPolynomial.C (HPow.hPow (↑p) n)))  …
  -/
  rw [map_mul, bind₁_C_right] at this
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    this : Eq (HMul.hMul ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i = …
    ⊢ Eq (HMul.hMul (wittStructureRat p Φ n) (MvPolynomial.C (HPow.hPow (↑p) n)))  …
  -/
  rw [wittStructureRat, this]; clear this
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => (MvPolynomial. …
  -/
  conv_lhs => simp only [map_sub, bind₁_X_right]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Eq (HSub.hSub ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod.mk i) …
  -/
  rw [sub_right_inj]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => (MvPolynomial. …
  -/
  simp only [map_sum, map_mul, bind₁_C_right, map_pow]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun x => HMul.hMul (HPow.hPow (MvPolynomial.C ↑p) x …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Write `wittStructureRat p φ n` in terms of `wittStructureRat p φ i` for `i < n`. -/
theorem wittStructureRat_rec (Φ : MvPolynomial idx ℚ) (n : ℕ) :
    wittStructureRat p Φ n =
      C (1 / (p : ℚ) ^ n) *
        (bind₁ (fun b => rename (fun i => (b, i)) (W_ ℚ n)) Φ -
          ∑ i ∈ range n, C ((p : ℚ) ^ i) * wittStructureRat p Φ i ^ p ^ (n - i)) := by
  calc
    wittStructureRat p Φ n = C (1 / (p : ℚ) ^ n) * (wittStructureRat p Φ n * C ((p : ℚ) ^ n)) := ?_
    _ = _ := by rw [wittStructureRat_rec_aux]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Eq (wittStructureRat p Φ n) (HMul.hMul (MvPolynomial.C (HDiv.hDiv 1 (HPow.hP …
  -/
  rw [mul_left_comm, ← C_mul, div_mul_cancel₀, C_1, mul_one]
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ Ne (HPow.hPow (↑p) n) 0
  -/
  exact pow_ne_zero _ (Nat.cast_ne_zero.2 hp.1.ne_zero)
  /-
    🎉 no goals
  -/


/-- `wittStructureInt Φ` is a family of polynomials `ℕ → MvPolynomial (idx × ℕ) ℤ`
that are uniquely characterised by the property that
```
bind₁ (wittStructureInt p Φ) (wittPolynomial p ℤ n) =
bind₁ (fun i ↦ (rename (prod.mk i) (wittPolynomial p ℤ n))) Φ
```
In other words: evaluating the `n`-th Witt polynomial on the family `wittStructureInt Φ`
is the same as evaluating `Φ` on the (appropriately renamed) `n`-th Witt polynomials.

See `wittStructureInt_prop` for this property,
and `wittStructureInt_existsUnique` for the fact that `wittStructureInt`
gives the unique family of polynomials with this property. -/
noncomputable def wittStructureInt (Φ : MvPolynomial idx ℤ) (n : ℕ) : MvPolynomial (idx × ℕ) ℤ :=
  Finsupp.mapRange Rat.num (Rat.num_intCast 0) (wittStructureRat p (map (Int.castRingHom ℚ) Φ) n)


theorem bind₁_rename_expand_wittPolynomial (Φ : MvPolynomial idx ℤ) (n : ℕ)
    (IH :
      ∀ m : ℕ,
        m < n + 1 →
          map (Int.castRingHom ℚ) (wittStructureInt p Φ m) =
            wittStructureRat p (map (Int.castRingHom ℚ) Φ) m) :
    bind₁ (fun b => rename (fun i => (b, i)) (expand p (W_ ℤ n))) Φ =
      bind₁ (fun i => expand p (wittStructureInt p Φ i)) (W_ ℤ n) := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    ⊢ Eq ((MvPolynomial.bind₁ fun b => (MvPolynomial.rename fun i => { fst := b, s …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((MvPolynomial.bind₁ fun b => ( …
  -/
  simp only [map_bind₁, map_rename, map_expand, rename_expand, map_wittPolynomial]
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.expand p) ((MvPolynomial.rena …
  -/
  have key := (wittStructureRat_prop p (map (Int.castRingHom ℚ) Φ) n).symm
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod.mk i)) (witt …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.expand p) ((MvPolynomial.rena …
  -/
  apply_fun expand p at key
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.expand p) ((MvPolynomial.bind₁ fun i => (MvPolynomial. …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.expand p) ((MvPolynomial.rena …
  -/
  simp only [expand_bind₁] at key
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.expand p) ((MvPolynomial. …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.expand p) ((MvPolynomial.rena …
  -/
  rw [key]; clear key
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.expand p) (wittStructureRat p …
  -/
  apply eval₂Hom_congr' rfl _ rfl
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    ⊢ ∀ (i : Nat), Membership.mem (wittPolynomial p Rat n).vars i → Membership.mem …
  -/
  rintro i hi -
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    i : Nat
    hi : Membership.mem (wittPolynomial p Rat n).vars i
    ⊢ Eq ((MvPolynomial.expand p) (wittStructureRat p ((MvPolynomial.map (Int.cast …
  -/
  rw [wittPolynomial_vars, Finset.mem_range] at hi
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq ((MvPolynomial.expand p) (wittStructureRat p ((MvPolynomial.map (Int.cast …
  -/
  simp only [IH i hi]
  /-
    🎉 no goals
  -/


theorem C_p_pow_dvd_bind₁_rename_wittPolynomial_sub_sum (Φ : MvPolynomial idx ℤ) (n : ℕ)
    (IH :
      ∀ m : ℕ,
        m < n →
          map (Int.castRingHom ℚ) (wittStructureInt p Φ m) =
            wittStructureRat p (map (Int.castRingHom ℚ) Φ) m) :
    (C ((p ^ n :) : ℤ) : MvPolynomial (idx × ℕ) ℤ) ∣
      bind₁ (fun b : idx => rename (fun i => (b, i)) (wittPolynomial p ℤ n)) Φ -
        ∑ i ∈ range n, C ((p : ℤ) ^ i) * wittStructureInt p Φ i ^ p ^ (n - i) := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    ⊢ Dvd.dvd (MvPolynomial.C ↑(HPow.hPow p n)) (HSub.hSub ((MvPolynomial.bind₁ fu …
  -/
  cases' n with n
  · simp only [isUnit_one, Int.ofNat_zero, Int.ofNat_succ, zero_add, pow_zero, C_1, IsUnit.dvd,
      Nat.cast_one]
  -- prepare a useful equation for rewriting
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    ⊢ Dvd.dvd (MvPolynomial.C ↑(HPow.hPow p (HAdd.hAdd n 1))) (HSub.hSub ((MvPolyn …
  -/
  have key := bind₁_rename_expand_wittPolynomial Φ n IH
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.bind₁ fun b => (MvPolynomial.rename fun i => { fst :=  …
    ⊢ Dvd.dvd (MvPolynomial.C ↑(HPow.hPow p (HAdd.hAdd n 1))) (HSub.hSub ((MvPolyn …
  -/
  apply_fun map (Int.castRingHom (ZMod (p ^ (n + 1)))) at key
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.map (Int.castRingHom (ZMod (HPow.hPow p (HAdd.hAdd n 1 …
    ⊢ Dvd.dvd (MvPolynomial.C ↑(HPow.hPow p (HAdd.hAdd n 1))) (HSub.hSub ((MvPolyn …
  -/
  conv_lhs at key => simp only [map_bind₁, map_rename, map_expand, map_wittPolynomial]
  -- clean up and massage
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename fun i_1 => { fst : …
    ⊢ Dvd.dvd (MvPolynomial.C ↑(HPow.hPow p (HAdd.hAdd n 1))) (HSub.hSub ((MvPolyn …
  -/
  rw [C_dvd_iff_zmod, RingHom.map_sub, sub_eq_zero, map_bind₁]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename fun i_1 => { fst : …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.map (Int.castRingHom (ZMod (H …
  -/
  simp only [map_rename, map_wittPolynomial, wittPolynomial_zmod_self]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → Eq ((MvPolynomial.map (Int.castRin …
    key : Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename fun i_1 => { fst : …
    ⊢ Eq ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename fun i_1 => { fst := i, …
  -/
  rw [key]; clear key IH
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom (ZMod (HPow.hPow p (HAdd.hAdd n 1)))) …
  -/
  rw [bind₁, aeval_wittPolynomial, map_sum, map_sum, Finset.sum_congr rfl]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq ((MvPolyno …
  -/
  intro k hk
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom (ZMod (HPow.hPow p (HAdd.hAdd n 1)))) …
  -/
  rw [Finset.mem_range, Nat.lt_succ_iff] at hk
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): was much slower
  -- simp only [← sub_eq_zero, ← RingHom.map_sub, ← C_dvd_iff_zmod, C_eq_coe_nat, ← mul_sub, ←
  --   Nat.cast_pow]
  rw [← sub_eq_zero, ← RingHom.map_sub, ← C_dvd_iff_zmod, C_eq_coe_nat, ← Nat.cast_pow,
    ← Nat.cast_pow, C_eq_coe_nat, ← mul_sub]
  have : p ^ (n + 1) = p ^ k * p ^ (n - k + 1) := by
    rw [← pow_add, ← add_assoc]; congr 2; rw [add_comm, ← tsub_eq_iff_eq_add_of_le hk]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : LE.le k n
    this : Eq (HPow.hPow p (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow p k) (HPow.hPow  …
    ⊢ Dvd.dvd (↑(HPow.hPow p (HAdd.hAdd n 1))) (HMul.hMul (↑(HPow.hPow p k)) (HSub …
  -/
  rw [this]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : LE.le k n
    this : Eq (HPow.hPow p (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow p k) (HPow.hPow  …
    ⊢ Dvd.dvd (↑(HMul.hMul (HPow.hPow p k) (HPow.hPow p (HAdd.hAdd (HSub.hSub n k) …
  -/
  rw [Nat.cast_mul, Nat.cast_pow, Nat.cast_pow]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : LE.le k n
    this : Eq (HPow.hPow p (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow p k) (HPow.hPow  …
    ⊢ Dvd.dvd (HMul.hMul (HPow.hPow (↑p) k) (HPow.hPow (↑p) (HAdd.hAdd (HSub.hSub  …
  -/
  apply mul_dvd_mul_left ((p : MvPolynomial (idx × ℕ) ℤ) ^ k)
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : LE.le k n
    this : Eq (HPow.hPow p (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow p k) (HPow.hPow  …
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd (HSub.hSub n k) 1)) (HSub.hSub (HPow.hPow …
  -/
  rw [show p ^ (n + 1 - k) = p * p ^ (n - k) by rw [← pow_succ', ← tsub_add_eq_add_tsub hk]]
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : LE.le k n
    this : Eq (HPow.hPow p (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow p k) (HPow.hPow  …
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd (HSub.hSub n k) 1)) (HSub.hSub (HPow.hPow …
  -/
  rw [pow_mul]
  -- the machine!
  /-
    case succ
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n k : Nat
    hk : LE.le k n
    this : Eq (HPow.hPow p (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow p k) (HPow.hPow  …
    ⊢ Dvd.dvd (HPow.hPow (↑p) (HAdd.hAdd (HSub.hSub n k) 1)) (HSub.hSub (HPow.hPow …
  -/
  apply dvd_sub_pow_of_dvd_sub
  rw [← C_eq_coe_nat, C_dvd_iff_zmod, RingHom.map_sub, sub_eq_zero, map_expand, RingHom.map_pow,
    MvPolynomial.expand_zmod]


@[simp]
theorem map_wittStructureInt (Φ : MvPolynomial idx ℤ) (n : ℕ) :
    map (Int.castRingHom ℚ) (wittStructureInt p Φ n) =
      wittStructureRat p (map (Int.castRingHom ℚ) Φ) n := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wittStructureInt p Φ n)) (witt …
  -/
  induction n using Nat.strong_induction_on with | h n IH => ?_
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wittStructureInt p Φ n)) (witt …
  -/
  rw [wittStructureInt, map_mapRange_eq_iff, Int.coe_castRingHom]
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    ⊢ ∀ (d : Finsupp (Prod idx Nat) Nat), Eq ((fun x => ↑x) (MvPolynomial.coeff d  …
  -/
  intro c
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    c : Finsupp (Prod idx Nat) Nat
    ⊢ Eq ((fun x => ↑x) (MvPolynomial.coeff c (wittStructureRat p ((MvPolynomial.m …
  -/
  rw [wittStructureRat_rec, coeff_C_mul, mul_comm, mul_div_assoc', mul_one]
  have sum_induction_steps :
      map (Int.castRingHom ℚ)
        (∑ i ∈ range n, C ((p : ℤ) ^ i) * wittStructureInt p Φ i ^ p ^ (n - i)) =
      ∑ i ∈ range n,
        C ((p : ℚ) ^ i) * wittStructureRat p (map (Int.castRingHom ℚ) Φ) i ^ p ^ (n - i) := by
    rw [map_sum]
    apply Finset.sum_congr rfl
    intro i hi
    rw [Finset.mem_range] at hi
    simp only [IH i hi, RingHom.map_mul, RingHom.map_pow, map_C]
    rfl
  simp only [← sum_induction_steps, ← map_wittPolynomial p (Int.castRingHom ℚ), ← map_rename, ←
    map_bind₁, ← RingHom.map_sub, coeff_map]
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    c : Finsupp (Prod idx Nat) Nat
    sum_induction_steps : Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((Finset.ra …
    ⊢ Eq (↑(HDiv.hDiv ((Int.castRingHom Rat) (MvPolynomial.coeff c (HSub.hSub ((Mv …
  -/
  rw [show (p : ℚ) ^ n = ((↑(p ^ n) : ℤ) : ℚ) by norm_cast]
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    c : Finsupp (Prod idx Nat) Nat
    sum_induction_steps : Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((Finset.ra …
    ⊢ Eq (↑(HDiv.hDiv ((Int.castRingHom Rat) (MvPolynomial.coeff c (HSub.hSub ((Mv …
  -/
  rw [← Rat.den_eq_one_iff, eq_intCast, Rat.den_div_intCast_eq_one_iff]
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    c : Finsupp (Prod idx Nat) Nat
    sum_induction_steps : Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((Finset.ra …
    ⊢ Dvd.dvd (↑(HPow.hPow p n)) (MvPolynomial.coeff c (HSub.hSub ((MvPolynomial.b …
  -/
  swap; · exact mod_cast pow_ne_zero n hp.1.ne_zero
          /-
            🎉 no goals
          -/
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    c : Finsupp (Prod idx Nat) Nat
    sum_induction_steps : Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((Finset.ra …
    ⊢ Dvd.dvd (↑(HPow.hPow p n)) (MvPolynomial.coeff c (HSub.hSub ((MvPolynomial.b …
  -/
  revert c; rw [← C_dvd_iff_dvd_coeff]
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq ((MvPolynomial.map (Int.castRingHom Rat)) (wi …
    sum_induction_steps : Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((Finset.ra …
    ⊢ Dvd.dvd (MvPolynomial.C ↑(HPow.hPow p n)) (HSub.hSub ((MvPolynomial.bind₁ fu …
  -/
  exact C_p_pow_dvd_bind₁_rename_wittPolynomial_sub_sum Φ n IH
  /-
    🎉 no goals
  -/


theorem wittStructureInt_prop (Φ : MvPolynomial idx ℤ) (n) :
    bind₁ (wittStructureInt p Φ) (wittPolynomial p ℤ n) =
      bind₁ (fun i => rename (Prod.mk i) (W_ ℤ n)) Φ := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ (wittStructureInt p Φ)) (wittPolynomial p Int n)) (( …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) ((MvPolynomial.bind₁ (wittStruc …
  -/
  have := wittStructureRat_prop p (map (Int.castRingHom ℚ) Φ) n
  simpa only [map_bind₁, ← eval₂Hom_map_hom, eval₂Hom_C_left, map_rename, map_wittPolynomial,
    AlgHom.coe_toRingHom, map_wittStructureInt]


theorem eq_wittStructureInt (Φ : MvPolynomial idx ℤ) (φ : ℕ → MvPolynomial (idx × ℕ) ℤ)
    (h : ∀ n, bind₁ φ (wittPolynomial p ℤ n) = bind₁ (fun i => rename (Prod.mk i) (W_ ℤ n)) Φ) :
    φ = wittStructureInt p Φ := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    φ : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
    ⊢ Eq φ (wittStructureInt p Φ)
  -/
  funext k
  /-
    case h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    φ : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
    k : Nat
    ⊢ Eq (φ k) (wittStructureInt p Φ k)
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  /-
    case h.a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    φ : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
    k : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) (φ k)) ((MvPolynomial.map (Int. …
  -/
  rw [map_wittStructureInt]
  -- Porting note: was `refine' congr_fun _ k`
  /-
    case h.a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    φ : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
    k : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom Rat)) (φ k)) (wittStructureRat p ((Mv …
  -/
  revert k
  /-
    case h.a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    φ : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
    ⊢ ∀ (k : Nat), Eq ((MvPolynomial.map (Int.castRingHom Rat)) (φ k)) (wittStruct …
  -/
  refine congr_fun ?_
  /-
    case h.a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    φ : Nat → MvPolynomial (Prod idx Nat) Int
    h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
    ⊢ Eq (fun k => (MvPolynomial.map (Int.castRingHom Rat)) (φ k)) (wittStructureR …
  -/
  apply ExistsUnique.unique (wittStructureRat_existsUnique p (map (Int.castRingHom ℚ) Φ))
    /-
      case h.a.py₁
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      φ : Nat → MvPolynomial (Prod idx Nat) Int
      h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
      ⊢ ∀ (n : Nat), Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.map (Int.castRin …
    -/
  · intro n
    /-
      case h.a.py₁
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      φ : Nat → MvPolynomial (Prod idx Nat) Int
      h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
      n : Nat
      ⊢ Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.map (Int.castRingHom Rat)) (φ …
    -/
    specialize h n
    /-
      case h.a.py₁
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      φ : Nat → MvPolynomial (Prod idx Nat) Int
      n : Nat
      h : Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPolynomial.bind₁  …
      ⊢ Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.map (Int.castRingHom Rat)) (φ …
    -/
    apply_fun map (Int.castRingHom ℚ) at h
    simpa only [map_bind₁, ← eval₂Hom_map_hom, eval₂Hom_C_left, map_rename, map_wittPolynomial,
      AlgHom.coe_toRingHom] using h
    /-
      case h.a.py₂
      p : Nat
      idx : Type u_2
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      φ : Nat → MvPolynomial (Prod idx Nat) Int
      h : ∀ (n : Nat), Eq ((MvPolynomial.bind₁ φ) (wittPolynomial p Int n)) ((MvPoly …
      ⊢ ∀ (n : Nat), Eq ((MvPolynomial.bind₁ (wittStructureRat p ((MvPolynomial.map  …
    -/
  · intro n; apply wittStructureRat_prop
             /-
               🎉 no goals
             -/


theorem wittStructureInt_existsUnique (Φ : MvPolynomial idx ℤ) :
    ∃! φ : ℕ → MvPolynomial (idx × ℕ) ℤ,
      ∀ n : ℕ,
        bind₁ φ (wittPolynomial p ℤ n) = bind₁ (fun i : idx => rename (Prod.mk i) (W_ ℤ n)) Φ :=
  ⟨wittStructureInt p Φ, wittStructureInt_prop _ _, eq_wittStructureInt _ _⟩


theorem witt_structure_prop (Φ : MvPolynomial idx ℤ) (n) :
    aeval (fun i => map (Int.castRingHom R) (wittStructureInt p Φ i)) (wittPolynomial p ℤ n) =
      aeval (fun i => rename (Prod.mk i) (W n)) Φ := by
  /-
    p : Nat
    R : Type u_1
    idx : Type u_2
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ Eq ((MvPolynomial.aeval fun i => (MvPolynomial.map (Int.castRingHom R)) (wit …
  -/
  convert congr_arg (map (Int.castRingHom R)) (wittStructureInt_prop p Φ n) using 1 <;>
      /-
        case h.e'_2
        p : Nat
        R : Type u_1
        idx : Type u_2
        inst✝ : CommRing R
        hp : Fact (Nat.Prime p)
        Φ : MvPolynomial idx Int
        n : Nat
        ⊢ Eq ((MvPolynomial.aeval fun i => (MvPolynomial.map (Int.castRingHom R)) (wit …
      -/
      rw [hom_bind₁] <;>
    /-
      case h.e'_2
      p : Nat
      R : Type u_1
      idx : Type u_2
      inst✝ : CommRing R
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      n : Nat
      ⊢ Eq ((MvPolynomial.aeval fun i => (MvPolynomial.map (Int.castRingHom R)) (wit …
    -/
    apply eval₂Hom_congr (RingHom.ext_int _ _) _ rfl
    /-
      p : Nat
      R : Type u_1
      idx : Type u_2
      inst✝ : CommRing R
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      n : Nat
      ⊢ Eq (fun i => (MvPolynomial.map (Int.castRingHom R)) (wittStructureInt p Φ i) …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      R : Type u_1
      idx : Type u_2
      inst✝ : CommRing R
      hp : Fact (Nat.Prime p)
      Φ : MvPolynomial idx Int
      n : Nat
      ⊢ Eq (fun i => (MvPolynomial.rename (Prod.mk i)) (wittPolynomial p R n)) fun i …
    -/
  · simp only [map_rename, map_wittPolynomial]
    /-
      🎉 no goals
    -/


theorem wittStructureInt_rename {σ : Type*} (Φ : MvPolynomial idx ℤ) (f : idx → σ) (n : ℕ) :
    wittStructureInt p (rename f Φ) n = rename (Prod.map f id) (wittStructureInt p Φ n) := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    σ : Type u_3
    Φ : MvPolynomial idx Int
    f : idx → σ
    n : Nat
    ⊢ Eq (wittStructureInt p ((MvPolynomial.rename f) Φ) n) ((MvPolynomial.rename  …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [map_rename, map_wittStructureInt, wittStructureRat, rename_bind₁, rename_rename,
    bind₁_rename]
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    σ : Type u_3
    Φ : MvPolynomial idx Int
    f : idx → σ
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ (Function.comp (fun i = …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittStructureRat_zero (Φ : MvPolynomial idx ℚ) :
    constantCoeff (wittStructureRat p Φ 0) = constantCoeff Φ := by
  simp only [wittStructureRat, bind₁, map_aeval, xInTermsOfW_zero, constantCoeff_rename,
    constantCoeff_wittPolynomial, aeval_X, constantCoeff_comp_algebraMap, eval₂Hom_zero'_apply,
    RingHom.id_apply]


theorem constantCoeff_wittStructureRat (Φ : MvPolynomial idx ℚ) (h : constantCoeff Φ = 0) (n : ℕ) :
    constantCoeff (wittStructureRat p Φ n) = 0 := by
  simp only [wittStructureRat, eval₂Hom_zero'_apply, h, bind₁, map_aeval, constantCoeff_rename,
    constantCoeff_wittPolynomial, constantCoeff_comp_algebraMap, RingHom.id_apply,
    constantCoeff_xInTermsOfW]


@[simp]
theorem constantCoeff_wittStructureInt_zero (Φ : MvPolynomial idx ℤ) :
    constantCoeff (wittStructureInt p Φ 0) = constantCoeff Φ := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    ⊢ Eq (MvPolynomial.constantCoeff (wittStructureInt p Φ 0)) (MvPolynomial.const …
  -/
  have inj : Function.Injective (Int.castRingHom ℚ) := by intro m n; exact Int.cast_inj.mp
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    inj : Function.Injective ⇑(Int.castRingHom Rat)
    ⊢ Eq (MvPolynomial.constantCoeff (wittStructureInt p Φ 0)) (MvPolynomial.const …
  -/
  apply inj
  rw [← constantCoeff_map, map_wittStructureInt, constantCoeff_wittStructureRat_zero,
    constantCoeff_map]


theorem constantCoeff_wittStructureInt (Φ : MvPolynomial idx ℤ) (h : constantCoeff Φ = 0) (n : ℕ) :
    constantCoeff (wittStructureInt p Φ n) = 0 := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    h : Eq (MvPolynomial.constantCoeff Φ) 0
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (wittStructureInt p Φ n)) 0
  -/
  have inj : Function.Injective (Int.castRingHom ℚ) := by intro m n; exact Int.cast_inj.mp
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    h : Eq (MvPolynomial.constantCoeff Φ) 0
    n : Nat
    inj : Function.Injective ⇑(Int.castRingHom Rat)
    ⊢ Eq (MvPolynomial.constantCoeff (wittStructureInt p Φ n)) 0
  -/
  apply inj
  /-
    case a
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    h : Eq (MvPolynomial.constantCoeff Φ) 0
    n : Nat
    inj : Function.Injective ⇑(Int.castRingHom Rat)
    ⊢ Eq ((Int.castRingHom Rat) (MvPolynomial.constantCoeff (wittStructureInt p Φ  …
  -/
  rw [← constantCoeff_map, map_wittStructureInt, constantCoeff_wittStructureRat, RingHom.map_zero]
  /-
    case a.h
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    Φ : MvPolynomial idx Int
    h : Eq (MvPolynomial.constantCoeff Φ) 0
    n : Nat
    inj : Function.Injective ⇑(Int.castRingHom Rat)
    ⊢ Eq (MvPolynomial.constantCoeff ((MvPolynomial.map (Int.castRingHom Rat)) Φ)) 0
  -/
  rw [constantCoeff_map, h, RingHom.map_zero]
  /-
    🎉 no goals
  -/


theorem wittStructureRat_vars [Fintype idx] (Φ : MvPolynomial idx ℚ) (n : ℕ) :
    (wittStructureRat p Φ n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ HasSubset.Subset (wittStructureRat p Φ n).vars (SProd.sprod Finset.univ (Fin …
  -/
  rw [wittStructureRat]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n : Nat
    ⊢ HasSubset.Subset ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i =>  …
  -/
  intro x hx
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n : Nat
    x : Prod idx Nat
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    ⊢ Membership.mem (SProd.sprod Finset.univ (Finset.range (HAdd.hAdd n 1))) x
  -/
  simp only [Finset.mem_product, true_and, Finset.mem_univ, Finset.mem_range]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n : Nat
    x : Prod idx Nat
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    ⊢ LT.lt x.2 (HAdd.hAdd n 1)
  -/
  obtain ⟨k, hk, hx'⟩ := mem_vars_bind₁ _ _ hx
  /-
    case intro.intro
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n : Nat
    x : Prod idx Nat
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    k : Nat
    hk : Membership.mem (xInTermsOfW p Rat n).vars k
    hx' : Membership.mem ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod. …
    ⊢ LT.lt x.2 (HAdd.hAdd n 1)
  -/
  obtain ⟨i, -, hx''⟩ := mem_vars_bind₁ _ _ hx'
  /-
    case intro.intro.intro.intro
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n : Nat
    x : Prod idx Nat
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    k : Nat
    hk : Membership.mem (xInTermsOfW p Rat n).vars k
    hx' : Membership.mem ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod. …
    i : idx
    hx'' : Membership.mem ((MvPolynomial.rename (Prod.mk i)) (wittPolynomial p Rat …
    ⊢ LT.lt x.2 (HAdd.hAdd n 1)
  -/
  obtain ⟨j, hj, rfl⟩ := mem_vars_rename _ _ hx''
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n k : Nat
    hk : Membership.mem (xInTermsOfW p Rat n).vars k
    i : idx
    j : Nat
    hj : Membership.mem (wittPolynomial p Rat k).vars j
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    hx' : Membership.mem ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod. …
    hx'' : Membership.mem ((MvPolynomial.rename (Prod.mk i)) (wittPolynomial p Rat …
    ⊢ LT.lt { fst := i, snd := j }.2 (HAdd.hAdd n 1)
  -/
  rw [wittPolynomial_vars, Finset.mem_range] at hj
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n k : Nat
    hk : Membership.mem (xInTermsOfW p Rat n).vars k
    i : idx
    j : Nat
    hj : LT.lt j (HAdd.hAdd k 1)
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    hx' : Membership.mem ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod. …
    hx'' : Membership.mem ((MvPolynomial.rename (Prod.mk i)) (wittPolynomial p Rat …
    ⊢ LT.lt { fst := i, snd := j }.2 (HAdd.hAdd n 1)
  -/
  replace hk := xInTermsOfW_vars_subset p _ hk
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n k : Nat
    i : idx
    j : Nat
    hj : LT.lt j (HAdd.hAdd k 1)
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    hx' : Membership.mem ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod. …
    hx'' : Membership.mem ((MvPolynomial.rename (Prod.mk i)) (wittPolynomial p Rat …
    hk : Membership.mem (Finset.range (HAdd.hAdd n 1)) k
    ⊢ LT.lt { fst := i, snd := j }.2 (HAdd.hAdd n 1)
  -/
  rw [Finset.mem_range] at hk
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Rat
    n k : Nat
    i : idx
    j : Nat
    hj : LT.lt j (HAdd.hAdd k 1)
    hx : Membership.mem ((MvPolynomial.bind₁ fun k => (MvPolynomial.bind₁ fun i => …
    hx' : Membership.mem ((MvPolynomial.bind₁ fun i => (MvPolynomial.rename (Prod. …
    hx'' : Membership.mem ((MvPolynomial.rename (Prod.mk i)) (wittPolynomial p Rat …
    hk : LT.lt k (HAdd.hAdd n 1)
    ⊢ LT.lt { fst := i, snd := j }.2 (HAdd.hAdd n 1)
  -/
  exact lt_of_lt_of_le hj hk
  /-
    🎉 no goals
  -/

-- we could relax the fintype on `idx`, but then we need to cast from finset to set.
-- for our applications `idx` is always finite.

theorem wittStructureInt_vars [Fintype idx] (Φ : MvPolynomial idx ℤ) (n : ℕ) :
    (wittStructureInt p Φ n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) := by
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Int
    n : Nat
    ⊢ HasSubset.Subset (wittStructureInt p Φ n).vars (SProd.sprod Finset.univ (Fin …
  -/
  have : Function.Injective (Int.castRingHom ℚ) := Int.cast_injective
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Int
    n : Nat
    this : Function.Injective ⇑(Int.castRingHom Rat)
    ⊢ HasSubset.Subset (wittStructureInt p Φ n).vars (SProd.sprod Finset.univ (Fin …
  -/
  rw [← vars_map_of_injective _ this, map_wittStructureInt]
  /-
    p : Nat
    idx : Type u_2
    hp : Fact (Nat.Prime p)
    inst✝ : Fintype idx
    Φ : MvPolynomial idx Int
    n : Nat
    this : Function.Injective ⇑(Int.castRingHom Rat)
    ⊢ HasSubset.Subset (wittStructureRat p ((MvPolynomial.map (Int.castRingHom Rat …
  -/
  apply wittStructureRat_vars
  /-
    🎉 no goals
  -/


