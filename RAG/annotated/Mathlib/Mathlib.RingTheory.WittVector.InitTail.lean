local notation "𝕎" => WittVector p



open scoped Classical in
/-- `WittVector.select P x`, for a predicate `P : ℕ → Prop` is the Witt vector
whose `n`-th coefficient is `x.coeff n` if `P n` is true, and `0` otherwise.
-/
def select (P : ℕ → Prop) (x : 𝕎 R) : 𝕎 R :=
  mk p fun n => if P n then x.coeff n else 0


open scoped Classical in
/-- The polynomial that witnesses that `WittVector.select` is a polynomial function.
`selectPoly n` is `X n` if `P n` holds, and `0` otherwise. -/
def selectPoly (n : ℕ) : MvPolynomial ℕ ℤ :=
  if P n then X n else 0


theorem coeff_select (x : 𝕎 R) (n : ℕ) :
    (select P x).coeff n = aeval x.coeff (selectPoly P n) := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    P : Nat → Prop
    x : WittVector p R
    n : Nat
    ⊢ Eq ((WittVector.select P x).coeff n) ((MvPolynomial.aeval x.coeff) (WittVect …
  -/
  dsimp [select, selectPoly]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    P : Nat → Prop
    x : WittVector p R
    n : Nat
    ⊢ Eq ((WittVector.mk p fun n => ite (P n) (x.coeff n) 0).coeff n) ((MvPolynomi …
  -/
  split_ifs with hi
    /-
      case pos
      p : Nat
      R : Type u_1
      inst✝ : CommRing R
      P : Nat → Prop
      x : WittVector p R
      n : Nat
      hi : P n
      ⊢ Eq ((WittVector.mk p fun n => ite (P n) (x.coeff n) 0).coeff n) ((MvPolynomi …
    -/
  · rw [aeval_X, mk]; simp only [hi, if_true]
                      /-
                        🎉 no goals
                      -/
    /-
      case neg
      p : Nat
      R : Type u_1
      inst✝ : CommRing R
      P : Nat → Prop
      x : WittVector p R
      n : Nat
      hi : Not (P n)
      ⊢ Eq ((WittVector.mk p fun n => ite (P n) (x.coeff n) 0).coeff n) ((MvPolynomi …
    -/
  · rw [map_zero, mk]; simp only [hi, if_false]
                       /-
                         🎉 no goals
                       -/

-- Porting note: replaced `@[is_poly]` with `instance`. Made the argument `P` implicit in doing so.

instance select_isPoly {P : ℕ → Prop} : IsPoly p fun _ _ x => select P x := by
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    P✝ P : Nat → Prop
    ⊢ WittVector.IsPoly p fun x x_1 x_2 => WittVector.select P x_2
  -/
  use selectPoly P
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    P✝ P : Nat → Prop
    ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq (WittVector.se …
  -/
  rintro R _Rcr x
  /-
    case h
    p n : Nat
    R✝ : Type u_1
    inst✝ : CommRing R✝
    P✝ P : Nat → Prop
    R : Type u_2
    _Rcr : CommRing R
    x : WittVector p R
    ⊢ Eq (WittVector.select P x).coeff fun n => (MvPolynomial.aeval x.coeff) (Witt …
  -/
  funext i
  /-
    case h.h
    p n : Nat
    R✝ : Type u_1
    inst✝ : CommRing R✝
    P✝ P : Nat → Prop
    R : Type u_2
    _Rcr : CommRing R
    x : WittVector p R
    i : Nat
    ⊢ Eq ((WittVector.select P x).coeff i) ((MvPolynomial.aeval x.coeff) (WittVect …
  -/
  apply coeff_select
  /-
    🎉 no goals
  -/


theorem select_add_select_not : ∀ x : 𝕎 R, select P x + select (fun i => ¬P i) x = x := by
  -- Porting note: TC search was insufficient to find this instance, even though all required
  -- instances exist. See zulip: [https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/WittVector.20saga/near/370073526]
  have : IsPoly p fun {R} [CommRing R] x ↦ select P x + select (fun i ↦ ¬P i) x :=
    IsPoly₂.diag (hf := IsPoly₂.comp)
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    ⊢ ∀ (x : WittVector p R), Eq (HAdd.hAdd (WittVector.select P x) (WittVector.se …
  -/
  ghost_calc x
  /-
    case refine_3
    p : Nat
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    ⊢ ∀ (n : Nat), Eq ((WittVector.ghostComponent n) (HAdd.hAdd (WittVector.select …
  -/
  intro n
  /-
    case refine_3
    p : Nat
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq ((WittVector.ghostComponent n) (HAdd.hAdd (WittVector.select P x) (WittVe …
  -/
  simp only [RingHom.map_add]
  suffices
    (bind₁ (selectPoly P)) (wittPolynomial p ℤ n) +
        (bind₁ (selectPoly fun i => ¬P i)) (wittPolynomial p ℤ n) =
      wittPolynomial p ℤ n by
    apply_fun aeval x.coeff at this
    simpa only [map_add, aeval_bind₁, ← coeff_select]
  simp only [wittPolynomial_eq_sum_C_mul_X_pow, selectPoly, map_sum, map_pow, map_mul,
    bind₁_X_right, bind₁_C_right, ← Finset.sum_add_distrib, ← mul_add]
  /-
    case refine_3
    p : Nat
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HPow.hPow (MvPoly …
  -/
  apply Finset.sum_congr rfl
  /-
    case refine_3
    p : Nat
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (HMul.hMul …
  -/
  refine fun m _ => mul_eq_mul_left_iff.mpr (Or.inl ?_)
  /-
    case refine_3
    p : Nat
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    n m : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) m
    ⊢ Eq (HAdd.hAdd (HPow.hPow (ite (P m) (MvPolynomial.X m) 0) (HPow.hPow p (HSub …
  -/
  rw [ite_pow, zero_pow (pow_ne_zero _ hp.out.ne_zero)]
  /-
    case refine_3
    p : Nat
    P : Nat → Prop
    hp : Fact (Nat.Prime p)
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    n m : Nat
    x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) m
    ⊢ Eq (HAdd.hAdd (ite (P m) (HPow.hPow (MvPolynomial.X m) (HPow.hPow p (HSub.hS …
  -/
  by_cases Pm : P m
    /-
      case pos
      p : Nat
      P : Nat → Prop
      hp : Fact (Nat.Prime p)
      this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
      R : Type u_1
      R._inst : CommRing R
      x : WittVector p R
      n m : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) m
      Pm : P m
      ⊢ Eq (HAdd.hAdd (ite (P m) (HPow.hPow (MvPolynomial.X m) (HPow.hPow p (HSub.hS …
    -/
  · rw [if_pos Pm, if_neg <| not_not_intro Pm, zero_pow Fin.pos'.ne', add_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      P : Nat → Prop
      hp : Fact (Nat.Prime p)
      this : WittVector.IsPoly p fun {R} [CommRing R] x => HAdd.hAdd (WittVector.sel …
      R : Type u_1
      R._inst : CommRing R
      x : WittVector p R
      n m : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) m
      Pm : Not (P m)
      ⊢ Eq (HAdd.hAdd (ite (P m) (HPow.hPow (MvPolynomial.X m) (HPow.hPow p (HSub.hS …
    -/
  · rwa [if_neg Pm, if_pos, zero_add]
    /-
      🎉 no goals
    -/


theorem coeff_add_of_disjoint (x y : 𝕎 R) (h : ∀ n, x.coeff n = 0 ∨ y.coeff n = 0) :
    (x + y).coeff n = x.coeff n + y.coeff n := by
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    x y : WittVector p R
    h : ∀ (n : Nat), Or (Eq (x.coeff n) 0) (Eq (y.coeff n) 0)
    ⊢ Eq ((HAdd.hAdd x y).coeff n) (HAdd.hAdd (x.coeff n) (y.coeff n))
  -/
  let P : ℕ → Prop := fun n => y.coeff n = 0
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    x y : WittVector p R
    h : ∀ (n : Nat), Or (Eq (x.coeff n) 0) (Eq (y.coeff n) 0)
    P : Nat → Prop := fun n => Eq (y.coeff n) 0
    ⊢ Eq ((HAdd.hAdd x y).coeff n) (HAdd.hAdd (x.coeff n) (y.coeff n))
  -/
  haveI : DecidablePred P := Classical.decPred P
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    hp : Fact (Nat.Prime p)
    x y : WittVector p R
    h : ∀ (n : Nat), Or (Eq (x.coeff n) 0) (Eq (y.coeff n) 0)
    P : Nat → Prop := fun n => Eq (y.coeff n) 0
    this : DecidablePred P
    ⊢ Eq ((HAdd.hAdd x y).coeff n) (HAdd.hAdd (x.coeff n) (y.coeff n))
  -/
  set z := mk p fun n => if P n then x.coeff n else y.coeff n
  have hx : select P z = x := by
    ext1 n; rw [select, coeff_mk, coeff_mk]
    split_ifs with hn
    · rfl
    · rw [(h n).resolve_right hn]
  have hy : select (fun i => ¬P i) z = y := by
    ext1 n; rw [select, coeff_mk, coeff_mk]
    split_ifs with hn
    · exact hn.symm
    · rfl
  calc
    (x + y).coeff n = z.coeff n := by rw [← hx, ← hy, select_add_select_not P z]
    _ = x.coeff n + y.coeff n := by
      simp only [z, mk.eq_1]
      split_ifs with y0
      · rw [y0, add_zero]
      · rw [h n |>.resolve_right y0, zero_add]


/-- `WittVector.init n x` is the Witt vector of which the first `n` coefficients are those from `x`
and all other coefficients are `0`.
See `WittVector.tail` for the complementary part.
-/
def init (n : ℕ) : 𝕎 R → 𝕎 R :=
  select fun i => i < n


/-- `WittVector.tail n x` is the Witt vector of which the first `n` coefficients are `0`
and all other coefficients are those from `x`.
See `WittVector.init` for the complementary part. -/
def tail (n : ℕ) : 𝕎 R → 𝕎 R :=
  select fun i => n ≤ i


@[simp]
theorem init_add_tail (x : 𝕎 R) (n : ℕ) : init n x + tail n x = x := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    n : Nat
    ⊢ Eq (HAdd.hAdd (WittVector.init n x) (WittVector.tail n x)) x
  -/
  simp only [init, tail, ← not_lt, select_add_select_not]
  /-
    🎉 no goals
  -/


/--
`init_ring` is an auxiliary tactic that discharges goals factoring `init` over ring operations.
-/
syntax (name := initRing) "init_ring" (" using " term)? : tactic

-- Porting note: this tactic requires that we turn hygiene off (note the free `n`).
-- TODO: make this tactic hygienic.

open Lean Elab Tactic in
elab_rules : tactic
| `(tactic| init_ring $[ using $a:term]?) => withMainContext <| set_option hygiene false in do
  evalTactic <|← `(tactic|(
    rw [WittVector.ext_iff]
    intro i
    simp only [WittVector.init, WittVector.select, WittVector.coeff_mk]
    split_ifs with hi <;> try {rfl}
    ))
  if let some e := a then
    evalTactic <|← `(tactic|(
      simp only [WittVector.add_coeff, WittVector.mul_coeff, WittVector.neg_coeff,
        WittVector.sub_coeff, WittVector.nsmul_coeff, WittVector.zsmul_coeff, WittVector.pow_coeff]
      apply MvPolynomial.eval₂Hom_congr' (RingHom.ext_int _ _) _ rfl
      rintro ⟨b, k⟩ h -
      replace h := $e:term p _ h
      simp only [Finset.mem_range, Finset.mem_product, true_and, Finset.mem_univ] at h
      have hk : k < n := by omega
      fin_cases b <;> simp only [Function.uncurry, Matrix.cons_val_zero, Matrix.head_cons,
        WittVector.coeff_mk, Matrix.cons_val_one, WittVector.mk, Fin.mk_zero, Matrix.cons_val',
        Matrix.empty_val', Matrix.cons_val_fin_one, Matrix.cons_val_zero,
        hk, if_true]
    ))

-- Porting note: `by init_ring` should suffice; this patches over an issue with `split_ifs`.
-- See zulip: [https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/.60split_ifs.60.20boxes.20itself.20into.20a.20corner]

@[simp]
theorem init_init (x : 𝕎 R) (n : ℕ) : init n (init n x) = init n x := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (WittVector.init n x)) (WittVector.init n x)
  -/
  rw [WittVector.ext_iff]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ ∀ (n_1 : Nat), Eq ((WittVector.init n (WittVector.init n x)).coeff n_1) ((Wi …
  -/
  intro i
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n i : Nat
    ⊢ Eq ((WittVector.init n (WittVector.init n x)).coeff i) ((WittVector.init n x …
  -/
  simp only [WittVector.init, WittVector.select, WittVector.coeff_mk]
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n i : Nat
    ⊢ Eq (ite (LT.lt i n) (ite (LT.lt i n) (x.coeff i) 0) 0) (ite (LT.lt i n) (x.c …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hi : i < n <;> simp [hi]
                          /-
                            🎉 no goals
                          -/


theorem init_add (x y : 𝕎 R) (n : ℕ) : init n (x + y) = init n (init n x + init n y) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (HAdd.hAdd x y)) (WittVector.init n (HAdd.hAdd (WittVe …
  -/
  init_ring using wittAdd_vars
  /-
    🎉 no goals
  -/


theorem init_mul (x y : 𝕎 R) (n : ℕ) : init n (x * y) = init n (init n x * init n y) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (HMul.hMul x y)) (WittVector.init n (HMul.hMul (WittVe …
  -/
  init_ring using wittMul_vars
  /-
    🎉 no goals
  -/


theorem init_neg (x : 𝕎 R) (n : ℕ) : init n (-x) = init n (-init n x) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (Neg.neg x)) (WittVector.init n (Neg.neg (WittVector.i …
  -/
  init_ring using wittNeg_vars
  /-
    🎉 no goals
  -/


theorem init_sub (x y : 𝕎 R) (n : ℕ) : init n (x - y) = init n (init n x - init n y) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (HSub.hSub x y)) (WittVector.init n (HSub.hSub (WittVe …
  -/
  init_ring using wittSub_vars
  /-
    🎉 no goals
  -/


theorem init_nsmul (m : ℕ) (x : 𝕎 R) (n : ℕ) : init n (m • x) = init n (m • init n x) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    x : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (HSMul.hSMul m x)) (WittVector.init n (HSMul.hSMul m ( …
  -/
  init_ring using fun p [Fact (Nat.Prime p)] n => wittNSMul_vars p m n
  /-
    🎉 no goals
  -/


theorem init_zsmul (m : ℤ) (x : 𝕎 R) (n : ℕ) : init n (m • x) = init n (m • init n x) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Int
    x : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (HSMul.hSMul m x)) (WittVector.init n (HSMul.hSMul m ( …
  -/
  init_ring using fun p [Fact (Nat.Prime p)] n => wittZSMul_vars p m n
  /-
    🎉 no goals
  -/


theorem init_pow (m : ℕ) (x : 𝕎 R) (n : ℕ) : init n (x ^ m) = init n (init n x ^ m) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    x : WittVector p R
    n : Nat
    ⊢ Eq (WittVector.init n (HPow.hPow x m)) (WittVector.init n (HPow.hPow (WittVe …
  -/
  init_ring using fun p [Fact (Nat.Prime p)] n => wittPow_vars p m n
  /-
    🎉 no goals
  -/


/-- `WittVector.init n x` is polynomial in the coefficients of `x`. -/
theorem init_isPoly (n : ℕ) : IsPoly p fun _ _ => init n :=
  select_isPoly (P := fun i => i < n)


