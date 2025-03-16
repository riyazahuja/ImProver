local notation "𝕎" => WittVector p -- type as `\bbW`


/-- A truncated Witt vector over `R` is a vector of elements of `R`,
i.e., the first `n` coefficients of a Witt vector.
We will define operations on this type that are compatible with the (untruncated) Witt
vector operations.

`TruncatedWittVector p n R` takes a parameter `p : ℕ` that is not used in the definition.
In practice, this number `p` is assumed to be a prime number,
and under this assumption we construct a ring structure on `TruncatedWittVector p n R`.
(`TruncatedWittVector p₁ n R` and `TruncatedWittVector p₂ n R` are definitionally
equal as types but will have different ring operations.)
-/
@[nolint unusedArguments]
def TruncatedWittVector (_ : ℕ) (n : ℕ) (R : Type*) :=
  Fin n → R


instance (p n : ℕ) (R : Type*) [Inhabited R] : Inhabited (TruncatedWittVector p n R) :=
  ⟨fun _ => default⟩


/-- Create a `TruncatedWittVector` from a vector `x`. -/
def mk (x : Fin n → R) : TruncatedWittVector p n R :=
  x


/-- `x.coeff i` is the `i`th entry of `x`. -/
def coeff (i : Fin n) (x : TruncatedWittVector p n R) : R :=
  x i


@[ext]
theorem ext {x y : TruncatedWittVector p n R} (h : ∀ i, x.coeff i = y.coeff i) : x = y :=
  funext h


@[simp]
theorem coeff_mk (x : Fin n → R) (i : Fin n) : (mk p x).coeff i = x i :=
  rfl


@[simp]
theorem mk_coeff (x : TruncatedWittVector p n R) : (mk p fun i => x.coeff i) = x := by
  /-
    p n : Nat
    R : Type u_1
    x : TruncatedWittVector p n R
    ⊢ Eq (TruncatedWittVector.mk p fun i => TruncatedWittVector.coeff i x) x
  -/
  ext i; rw [coeff_mk]
         /-
           🎉 no goals
         -/


/-- We can turn a truncated Witt vector `x` into a Witt vector
by setting all coefficients after `x` to be 0.
-/
def out (x : TruncatedWittVector p n R) : 𝕎 R :=
  @WittVector.mk' p _ fun i => if h : i < n then x.coeff ⟨i, h⟩ else 0


@[simp]
theorem coeff_out (x : TruncatedWittVector p n R) (i : Fin n) : x.out.coeff i = x.coeff i := by
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : TruncatedWittVector p n R
    i : Fin n
    ⊢ Eq (x.out.coeff ↑i) (TruncatedWittVector.coeff i x)
  -/
  rw [out]; dsimp only; rw [dif_pos i.is_lt, Fin.eta]
                        /-
                          🎉 no goals
                        -/


theorem out_injective : Injective (@out p n R _) := by
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Function.Injective TruncatedWittVector.out
  -/
  intro x y h
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x y : TruncatedWittVector p n R
    h : Eq x.out y.out
    ⊢ Eq x y
  -/
  ext i
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x y : TruncatedWittVector p n R
    h : Eq x.out y.out
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i x) (TruncatedWittVector.coeff i y)
  -/
  rw [WittVector.ext_iff] at h
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x y : TruncatedWittVector p n R
    h : ∀ (n_1 : Nat), Eq (x.out.coeff n_1) (y.out.coeff n_1)
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i x) (TruncatedWittVector.coeff i y)
  -/
  simpa only [coeff_out] using h ↑i
  /-
    🎉 no goals
  -/


/-- `truncateFun n x` uses the first `n` entries of `x` to construct a `TruncatedWittVector`,
which has the same base `p` as `x`.
This function is bundled into a ring homomorphism in `WittVector.truncate` -/
def truncateFun (x : 𝕎 R) : TruncatedWittVector p n R :=
  TruncatedWittVector.mk p fun i => x.coeff i


@[simp]
theorem coeff_truncateFun (x : 𝕎 R) (i : Fin n) : (truncateFun n x).coeff i = x.coeff i := by
  /-
    p n : Nat
    R : Type u_1
    x : WittVector p R
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i (WittVector.truncateFun n x)) (x.coeff ↑i)
  -/
  rw [truncateFun, TruncatedWittVector.coeff_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem out_truncateFun (x : 𝕎 R) : (truncateFun n x).out = init n x := by
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    ⊢ Eq (WittVector.truncateFun n x).out (WittVector.init n x)
  -/
  ext i
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    i : Nat
    ⊢ Eq ((WittVector.truncateFun n x).out.coeff i) ((WittVector.init n x).coeff i)
  -/
  dsimp [TruncatedWittVector.out, init, select, coeff_mk]
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    i : Nat
    ⊢ Eq (dite (LT.lt i n) (fun h => TruncatedWittVector.coeff ⟨i, h⟩ (WittVector. …
  -/
  split_ifs with hi; swap; · rfl
                             /-
                               🎉 no goals
                             -/
  /-
    case pos
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (TruncatedWittVector.coeff ⟨i, hi⟩ (WittVector.truncateFun n x)) (x.coeff …
  -/
  rw [coeff_truncateFun, Fin.val_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem truncateFun_out (x : TruncatedWittVector p n R) : x.out.truncateFun n = x := by
  /-
    p n : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : TruncatedWittVector p n R
    ⊢ Eq (WittVector.truncateFun n x.out) x
  -/
  simp only [WittVector.truncateFun, coeff_out, mk_coeff]
  /-
    🎉 no goals
  -/


instance : Zero (TruncatedWittVector p n R) :=
  ⟨truncateFun n 0⟩


instance : One (TruncatedWittVector p n R) :=
  ⟨truncateFun n 1⟩


instance : NatCast (TruncatedWittVector p n R) :=
  ⟨fun i => truncateFun n i⟩


instance : IntCast (TruncatedWittVector p n R) :=
  ⟨fun i => truncateFun n i⟩


instance : Add (TruncatedWittVector p n R) :=
  ⟨fun x y => truncateFun n (x.out + y.out)⟩


instance : Mul (TruncatedWittVector p n R) :=
  ⟨fun x y => truncateFun n (x.out * y.out)⟩


instance : Neg (TruncatedWittVector p n R) :=
  ⟨fun x => truncateFun n (-x.out)⟩


instance : Sub (TruncatedWittVector p n R) :=
  ⟨fun x y => truncateFun n (x.out - y.out)⟩


instance hasNatScalar : SMul ℕ (TruncatedWittVector p n R) :=
  ⟨fun m x => truncateFun n (m • x.out)⟩


instance hasIntScalar : SMul ℤ (TruncatedWittVector p n R) :=
  ⟨fun m x => truncateFun n (m • x.out)⟩


instance hasNatPow : Pow (TruncatedWittVector p n R) ℕ :=
  ⟨fun x m => truncateFun n (x.out ^ m)⟩


@[simp]
theorem coeff_zero (i : Fin n) : (0 : TruncatedWittVector p n R).coeff i = 0 := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i 0) 0
  -/
  show coeff i (truncateFun _ 0 : TruncatedWittVector p n R) = 0
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i (WittVector.truncateFun n 0)) 0
  -/
  rw [coeff_truncateFun, WittVector.zero_coeff]
  /-
    🎉 no goals
  -/


/-- A macro tactic used to prove that `truncateFun` respects ring operations. -/
macro (name := witt_truncateFun_tac) "witt_truncateFun_tac" : tactic =>
  `(tactic|
    { show _ = WittVector.truncateFun n _
      apply TruncatedWittVector.out_injective
      iterate rw [WittVector.out_truncateFun]
      first
      | rw [WittVector.init_add]
      | rw [WittVector.init_mul]
      | rw [WittVector.init_neg]
      | rw [WittVector.init_sub]
      | rw [WittVector.init_nsmul]
      | rw [WittVector.init_zsmul]
      | rw [WittVector.init_pow]})


theorem truncateFun_surjective : Surjective (@truncateFun p n R) :=
  Function.RightInverse.surjective TruncatedWittVector.truncateFun_out


@[simp]
theorem truncateFun_zero : truncateFun n (0 : 𝕎 R) = 0 := rfl


@[simp]
theorem truncateFun_one : truncateFun n (1 : 𝕎 R) = 1 := rfl


@[simp]
theorem truncateFun_add (x y : 𝕎 R) :
    truncateFun n (x + y) = truncateFun n x + truncateFun n y := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    ⊢ Eq (WittVector.truncateFun n (HAdd.hAdd x y)) (HAdd.hAdd (WittVector.truncat …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


@[simp]
theorem truncateFun_mul (x y : 𝕎 R) :
    truncateFun n (x * y) = truncateFun n x * truncateFun n y := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    ⊢ Eq (WittVector.truncateFun n (HMul.hMul x y)) (HMul.hMul (WittVector.truncat …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


theorem truncateFun_neg (x : 𝕎 R) : truncateFun n (-x) = -truncateFun n x := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    ⊢ Eq (WittVector.truncateFun n (Neg.neg x)) (Neg.neg (WittVector.truncateFun n …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


theorem truncateFun_sub (x y : 𝕎 R) :
    truncateFun n (x - y) = truncateFun n x - truncateFun n y := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x y : WittVector p R
    ⊢ Eq (WittVector.truncateFun n (HSub.hSub x y)) (HSub.hSub (WittVector.truncat …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


theorem truncateFun_nsmul (m : ℕ) (x : 𝕎 R) : truncateFun n (m • x) = m • truncateFun n x := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    x : WittVector p R
    ⊢ Eq (WittVector.truncateFun n (HSMul.hSMul m x)) (HSMul.hSMul m (WittVector.t …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


theorem truncateFun_zsmul (m : ℤ) (x : 𝕎 R) : truncateFun n (m • x) = m • truncateFun n x := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Int
    x : WittVector p R
    ⊢ Eq (WittVector.truncateFun n (HSMul.hSMul m x)) (HSMul.hSMul m (WittVector.t …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


theorem truncateFun_pow (x : 𝕎 R) (m : ℕ) : truncateFun n (x ^ m) = truncateFun n x ^ m := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    m : Nat
    ⊢ Eq (WittVector.truncateFun n (HPow.hPow x m)) (HPow.hPow (WittVector.truncat …
  -/
  witt_truncateFun_tac
  /-
    🎉 no goals
  -/


theorem truncateFun_natCast (m : ℕ) : truncateFun n (m : 𝕎 R) = m := rfl


@[deprecated (since := "2024-04-17")]
alias truncateFun_nat_cast := truncateFun_natCast


theorem truncateFun_intCast (m : ℤ) : truncateFun n (m : 𝕎 R) = m := rfl


@[deprecated (since := "2024-04-17")]
alias truncateFun_int_cast := truncateFun_intCast


instance instCommRing : CommRing (TruncatedWittVector p n R) :=
  (truncateFun_surjective p n R).commRing _ (truncateFun_zero p n R) (truncateFun_one p n R)
    (truncateFun_add n) (truncateFun_mul n) (truncateFun_neg n) (truncateFun_sub n)
    (truncateFun_nsmul n) (truncateFun_zsmul n) (truncateFun_pow n) (truncateFun_natCast n)
    (truncateFun_intCast n)


/-- `truncate n` is a ring homomorphism that truncates `x` to its first `n` entries
to obtain a `TruncatedWittVector`, which has the same base `p` as `x`. -/
noncomputable def truncate : 𝕎 R →+* TruncatedWittVector p n R where
  toFun := truncateFun n
  map_zero' := truncateFun_zero p n R
  map_add' := truncateFun_add n
  map_one' := truncateFun_one p n R
  map_mul' := truncateFun_mul n


theorem truncate_surjective : Surjective (truncate n : 𝕎 R → TruncatedWittVector p n R) :=
  truncateFun_surjective p n R


@[simp]
theorem coeff_truncate (x : 𝕎 R) (i : Fin n) : (truncate n x).coeff i = x.coeff i :=
  coeff_truncateFun _ _


theorem mem_ker_truncate (x : 𝕎 R) :
    x ∈ RingHom.ker (truncate (p := p) n) ↔ ∀ i < n, x.coeff i = 0 := by
  simp only [RingHom.mem_ker, truncate, truncateFun, RingHom.coe_mk, TruncatedWittVector.ext_iff,
    TruncatedWittVector.coeff_mk, coeff_zero]
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    ⊢ Iff (∀ (i : Fin n), Eq (TruncatedWittVector.coeff i ({ toFun := WittVector.t …
  -/
  exact Fin.forall_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem truncate_mk' (f : ℕ → R) :
    truncate n (@mk' p _ f) = TruncatedWittVector.mk _ fun k => f k := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    f : Nat → R
    ⊢ Eq ((WittVector.truncate n) { coeff := f }) (TruncatedWittVector.mk p fun k  …
  -/
  ext i
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    f : Nat → R
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i ((WittVector.truncate n) { coeff := f })) (T …
  -/
  simp only [coeff_truncate, TruncatedWittVector.coeff_mk]
  /-
    🎉 no goals
  -/


/-- A ring homomorphism that truncates a truncated Witt vector of length `m` to
a truncated Witt vector of length `n`, for `n ≤ m`.
-/
def truncate {m : ℕ} (hm : n ≤ m) : TruncatedWittVector p m R →+* TruncatedWittVector p n R :=
  RingHom.liftOfRightInverse (WittVector.truncate m) out truncateFun_out
    ⟨WittVector.truncate n, by
      /-
        p n : Nat
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : Fact (Nat.Prime p)
        m : Nat
        hm : LE.le n m
        ⊢ LE.le (RingHom.ker (WittVector.truncate m)) (RingHom.ker (WittVector.truncat …
      -/
      intro x
      /-
        p n : Nat
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : Fact (Nat.Prime p)
        m : Nat
        hm : LE.le n m
        x : WittVector p R
        ⊢ Membership.mem (RingHom.ker (WittVector.truncate m)) x → Membership.mem (Rin …
      -/
      simp only [WittVector.mem_ker_truncate]
      /-
        p n : Nat
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : Fact (Nat.Prime p)
        m : Nat
        hm : LE.le n m
        x : WittVector p R
        ⊢ (∀ (i : Nat), LT.lt i m → Eq (x.coeff i) 0) → ∀ (i : Nat), LT.lt i n → Eq (x …
      -/
      intro h i hi
      /-
        p n : Nat
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : Fact (Nat.Prime p)
        m : Nat
        hm : LE.le n m
        x : WittVector p R
        h : ∀ (i : Nat), LT.lt i m → Eq (x.coeff i) 0
        i : Nat
        hi : LT.lt i n
        ⊢ Eq (x.coeff i) 0
      -/
      exact h i (lt_of_lt_of_le hi hm)⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem truncate_comp_wittVector_truncate {m : ℕ} (hm : n ≤ m) :
    (truncate (p := p) (R := R) hm).comp (WittVector.truncate m) = WittVector.truncate n :=
  RingHom.liftOfRightInverse_comp _ _ _ _


@[simp]
theorem truncate_wittVector_truncate {m : ℕ} (hm : n ≤ m) (x : 𝕎 R) :
    truncate hm (WittVector.truncate m x) = WittVector.truncate n x :=
  RingHom.liftOfRightInverse_comp_apply _ _ _ _ _


@[simp]
theorem truncate_truncate {n₁ n₂ n₃ : ℕ} (h1 : n₁ ≤ n₂) (h2 : n₂ ≤ n₃)
    (x : TruncatedWittVector p n₃ R) :
    (truncate h1) (truncate h2 x) = truncate (h1.trans h2) x := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    n₁ n₂ n₃ : Nat
    h1 : LE.le n₁ n₂
    h2 : LE.le n₂ n₃
    x : TruncatedWittVector p n₃ R
    ⊢ Eq ((TruncatedWittVector.truncate h1) ((TruncatedWittVector.truncate h2) x)) …
  -/
  obtain ⟨x, rfl⟩ := WittVector.truncate_surjective (p := p) n₃ R x
  /-
    case intro
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    n₁ n₂ n₃ : Nat
    h1 : LE.le n₁ n₂
    h2 : LE.le n₂ n₃
    x : WittVector p R
    ⊢ Eq ((TruncatedWittVector.truncate h1) ((TruncatedWittVector.truncate h2) ((W …
  -/
  simp only [truncate_wittVector_truncate]
  /-
    🎉 no goals
  -/


@[simp]
theorem truncate_comp {n₁ n₂ n₃ : ℕ} (h1 : n₁ ≤ n₂) (h2 : n₂ ≤ n₃) :
    (truncate (p := p) (R := R) h1).comp (truncate h2) = truncate (h1.trans h2) := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    n₁ n₂ n₃ : Nat
    h1 : LE.le n₁ n₂
    h2 : LE.le n₂ n₃
    ⊢ Eq ((TruncatedWittVector.truncate h1).comp (TruncatedWittVector.truncate h2) …
  -/
  ext1 x; simp only [truncate_truncate, Function.comp_apply, RingHom.coe_comp]
          /-
            🎉 no goals
          -/


theorem truncate_surjective {m : ℕ} (hm : n ≤ m) : Surjective (truncate (p := p) (R := R) hm) := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    hm : LE.le n m
    ⊢ Function.Surjective ⇑(TruncatedWittVector.truncate hm)
  -/
  intro x
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    hm : LE.le n m
    x : TruncatedWittVector p n R
    ⊢ Exists fun a => Eq ((TruncatedWittVector.truncate hm) a) x
  -/
  obtain ⟨x, rfl⟩ := WittVector.truncate_surjective (p := p) _ R x
  /-
    case intro
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    hm : LE.le n m
    x : WittVector p R
    ⊢ Exists fun a => Eq ((TruncatedWittVector.truncate hm) a) ((WittVector.trunca …
  -/
  exact ⟨WittVector.truncate _ x, truncate_wittVector_truncate _ _⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_truncate {m : ℕ} (hm : n ≤ m) (i : Fin n) (x : TruncatedWittVector p m R) :
    (truncate hm x).coeff i = x.coeff (Fin.castLE hm i) := by
  /-
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    hm : LE.le n m
    i : Fin n
    x : TruncatedWittVector p m R
    ⊢ Eq (TruncatedWittVector.coeff i ((TruncatedWittVector.truncate hm) x)) (Trun …
  -/
  obtain ⟨y, rfl⟩ := @WittVector.truncate_surjective p _ _ _ _ x
  /-
    case intro
    p n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    m : Nat
    hm : LE.le n m
    i : Fin n
    y : WittVector p R
    ⊢ Eq (TruncatedWittVector.coeff i ((TruncatedWittVector.truncate hm) ((WittVec …
  -/
  simp only [truncate_wittVector_truncate, WittVector.coeff_truncate, Fin.coe_castLE]
  /-
    🎉 no goals
  -/


instance {R : Type*} [Fintype R] : Fintype (TruncatedWittVector p n R) :=
  Pi.instFintype


theorem card {R : Type*} [Fintype R] :
    Fintype.card (TruncatedWittVector p n R) = Fintype.card R ^ n := by
  /-
    p n : Nat
    R : Type u_2
    inst✝ : Fintype R
    ⊢ Eq (Fintype.card (TruncatedWittVector p n R)) (HPow.hPow (Fintype.card R) n)
  -/
  simp only [TruncatedWittVector, Fintype.card_fin, Fintype.card_fun]
  /-
    🎉 no goals
  -/


theorem iInf_ker_truncate : ⨅ i : ℕ, RingHom.ker (WittVector.truncate (p := p) (R := R) i) = ⊥ := by
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    ⊢ Eq (iInf fun i => RingHom.ker (WittVector.truncate i)) Bot.bot
  -/
  rw [Submodule.eq_bot_iff]
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    ⊢ ∀ (x : WittVector p R), Membership.mem (iInf fun i => RingHom.ker (WittVecto …
  -/
  intro x hx
  /-
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    hx : Membership.mem (iInf fun i => RingHom.ker (WittVector.truncate i)) x
    ⊢ Eq x 0
  -/
  ext
  /-
    case h
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    hx : Membership.mem (iInf fun i => RingHom.ker (WittVector.truncate i)) x
    n✝ : Nat
    ⊢ Eq (x.coeff n✝) (WittVector.coeff 0 n✝)
  -/
  simp only [WittVector.mem_ker_truncate, Ideal.mem_iInf, WittVector.zero_coeff] at hx ⊢
  /-
    case h
    p : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Fact (Nat.Prime p)
    x : WittVector p R
    n✝ : Nat
    hx : ∀ (i i_1 : Nat), LT.lt i_1 i → Eq (x.coeff i_1) 0
    ⊢ Eq (x.coeff n✝) 0
  -/
  exact hx _ _ (Nat.lt_succ_self _)
  /-
    🎉 no goals
  -/


/-- Given a family `fₖ : S → TruncatedWittVector p k R` and `s : S`, we produce a Witt vector by
defining the `k`th entry to be the final entry of `fₖ s`.
-/
def liftFun (s : S) : 𝕎 R :=
  @WittVector.mk' p _ fun k => TruncatedWittVector.coeff (Fin.last k) (f (k + 1) s)


include f_compat in
@[simp]
theorem truncate_liftFun (s : S) : WittVector.truncate n (liftFun f s) = f n s := by
  /-
    p n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    s : S
    ⊢ Eq ((WittVector.truncate n) (WittVector.liftFun f s)) ((f n) s)
  -/
  ext i
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    s : S
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff i ((WittVector.truncate n) (WittVector.liftFun …
  -/
  simp only [liftFun, TruncatedWittVector.coeff_mk, WittVector.truncate_mk']
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    s : S
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff (Fin.last ↑i) ((f (HAdd.hAdd (↑i) 1)) s)) (Tru …
  -/
  rw [← f_compat (i + 1) n i.is_lt, RingHom.comp_apply, TruncatedWittVector.coeff_truncate]
  /-
    case h
    p n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    s : S
    i : Fin n
    ⊢ Eq (TruncatedWittVector.coeff (Fin.castLE ⋯ (Fin.last ↑i)) ((f n) s)) (Trunc …
  -/
  congr 1 with _
  /-
    🎉 no goals
  -/


/--
Given compatible ring homs from `S` into `TruncatedWittVector n` for each `n`, we can lift these
to a ring hom `S → 𝕎 R`.

`lift` defines the universal property of `𝕎 R` as the inverse limit of `TruncatedWittVector n`.
-/
def lift : S →+* 𝕎 R := by
  refine {  toFun := liftFun f
            map_zero' := ?_
            map_one' := ?_
            map_add' := ?_
            map_mul' := ?_ } <;>
    /-
      case refine_1
      p n : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : Fact (Nat.Prime p)
      S : Type u_2
      inst✝ : Semiring S
      f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
      f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
      ⊢ Eq (WittVector.liftFun f 1) 1
    -/
    /-
      case refine_1
      p n : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : Fact (Nat.Prime p)
      S : Type u_2
      inst✝ : Semiring S
      f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
      f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
      ⊢ Eq (WittVector.liftFun f 1) 1
    -/
    /-
      case refine_1
      p n : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : Fact (Nat.Prime p)
      S : Type u_2
      inst✝ : Semiring S
      f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
      f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
      ⊢ Eq (WittVector.liftFun f 1) 1
    -/
    /-
      case refine_1
      p n : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : Fact (Nat.Prime p)
      S : Type u_2
      inst✝ : Semiring S
      f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
      f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
      ⊢ ∀ (i : Nat), Membership.mem (RingHom.ker (WittVector.truncate i)) (HSub.hSub …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      p n : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : Fact (Nat.Prime p)
      S : Type u_2
      inst✝ : Semiring S
      f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
      f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
      x✝ y✝ : S
      ⊢ Eq (WittVector.liftFun f (HAdd.hAdd x✝ y✝)) (HAdd.hAdd (WittVector.liftFun f …
    -/
    rw [← sub_eq_zero, ← Ideal.mem_bot, ← iInf_ker_truncate, Ideal.mem_iInf]
    /-
      case refine_4
      p n : Nat
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : Fact (Nat.Prime p)
      S : Type u_2
      inst✝ : Semiring S
      f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
      f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
      x✝ y✝ : S
      ⊢ ∀ (i : Nat), Membership.mem (RingHom.ker (WittVector.truncate i)) (HSub.hSub …
    -/
    simp [RingHom.mem_ker, f_compat])
    /-
      🎉 no goals
    -/


@[simp]
theorem truncate_lift (s : S) : WittVector.truncate n (lift _ f_compat s) = f n s :=
  truncate_liftFun _ f_compat s


@[simp]
theorem truncate_comp_lift : (WittVector.truncate n).comp (lift _ f_compat) = f n := by
  /-
    p n : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    ⊢ Eq ((WittVector.truncate n).comp (WittVector.lift f f_compat)) (f n)
  -/
  ext1; rw [RingHom.comp_apply, truncate_lift]
        /-
          🎉 no goals
        -/


/-- The uniqueness part of the universal property of `𝕎 R`. -/
theorem lift_unique (g : S →+* 𝕎 R) (g_compat : ∀ k, (WittVector.truncate k).comp g = f k) :
    lift _ f_compat = g := by
  /-
    p : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    g : RingHom S (WittVector p R)
    g_compat : ∀ (k : Nat), Eq ((WittVector.truncate k).comp g) (f k)
    ⊢ Eq (WittVector.lift f f_compat) g
  -/
  ext1 x
  /-
    case a
    p : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    g : RingHom S (WittVector p R)
    g_compat : ∀ (k : Nat), Eq ((WittVector.truncate k).comp g) (f k)
    x : S
    ⊢ Eq ((WittVector.lift f f_compat) x) (g x)
  -/
  rw [← sub_eq_zero, ← Ideal.mem_bot, ← iInf_ker_truncate, Ideal.mem_iInf]
  /-
    case a
    p : Nat
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : Fact (Nat.Prime p)
    S : Type u_2
    inst✝ : Semiring S
    f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
    f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
    g : RingHom S (WittVector p R)
    g_compat : ∀ (k : Nat), Eq ((WittVector.truncate k).comp g) (f k)
    x : S
    ⊢ ∀ (i : Nat), Membership.mem (RingHom.ker (WittVector.truncate i)) (HSub.hSub …
  -/
  intro i
  simp only [RingHom.mem_ker, g_compat, ← RingHom.comp_apply, truncate_comp_lift, RingHom.map_sub,
    sub_self]


/-- The universal property of `𝕎 R` as projective limit of truncated Witt vector rings. -/
@[simps]
def liftEquiv : { f : ∀ k, S →+* TruncatedWittVector p k R // ∀ (k₁ k₂) (hk : k₁ ≤ k₂),
    (TruncatedWittVector.truncate hk).comp (f k₂) = f k₁ } ≃ (S →+* 𝕎 R) where
  toFun f := lift f.1 f.2
  invFun g :=
    ⟨fun k => (truncate k).comp g, by
      /-
        p n : Nat
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : Fact (Nat.Prime p)
        S : Type u_2
        inst✝ : Semiring S
        f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
        f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
        g : RingHom S (WittVector p R)
        ⊢ ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.truncate hk).co …
      -/
      intro _ _ h
      /-
        p n : Nat
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : Fact (Nat.Prime p)
        S : Type u_2
        inst✝ : Semiring S
        f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
        f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
        g : RingHom S (WittVector p R)
        k₁✝ k₂✝ : Nat
        h : LE.le k₁✝ k₂✝
        ⊢ Eq ((TruncatedWittVector.truncate h).comp ((fun k => (WittVector.truncate k) …
      -/
      simp only [← RingHom.comp_assoc, truncate_comp_wittVector_truncate]⟩
      /-
        🎉 no goals
      -/
                 /-
                   p n : Nat
                   R : Type u_1
                   inst✝² : CommRing R
                   inst✝¹ : Fact (Nat.Prime p)
                   S : Type u_2
                   inst✝ : Semiring S
                   f : (k : Nat) → RingHom S (TruncatedWittVector p k R)
                   f_compat : ∀ (k₁ k₂ : Nat) (hk : LE.le k₁ k₂), Eq ((TruncatedWittVector.trunca …
                   ⊢ Function.LeftInverse (fun g => ⟨fun k => (WittVector.truncate k).comp g, ⋯⟩) …
                 -/
  left_inv := by rintro ⟨f, hf⟩; simp only [truncate_comp_lift]
                                 /-
                                   🎉 no goals
                                 -/
  right_inv _ := lift_unique _ _ fun _ => rfl


theorem hom_ext (g₁ g₂ : S →+* 𝕎 R) (h : ∀ k, (truncate k).comp g₁ = (truncate k).comp g₂) :
    g₁ = g₂ :=
  liftEquiv.symm.injective <| Subtype.ext <| funext h


