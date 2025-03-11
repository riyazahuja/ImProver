/-- We can partition a finite family into `partition_card ε` sets, such that the remainders
in each set are close together. -/
theorem exists_partition_int (n : ℕ) {ε : ℝ} (hε : 0 < ε) {b : ℤ} (hb : b ≠ 0) (A : Fin n → ℤ) :
    ∃ t : Fin n → Fin ⌈1 / ε⌉₊,
    ∀ i₀ i₁, t i₀ = t i₁ → ↑(abs (A i₁ % b - A i₀ % b)) < abs b • ε := by
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    ⊢ Exists fun t => ∀ (i₀ i₁ : Fin n), Eq (t i₀) (t i₁) → LT.lt (↑(abs (HSub.hSu …
  -/
  have hb' : (0 : ℝ) < ↑(abs b) := Int.cast_pos.mpr (abs_pos.mpr hb)
  have hbε : 0 < abs b • ε := by
    rw [Algebra.smul_def]
    exact mul_pos hb' hε
  have hfloor : ∀ i, 0 ≤ floor ((A i % b : ℤ) / abs b • ε : ℝ) :=
    fun _ ↦ floor_nonneg.mpr (div_nonneg (cast_nonneg.mpr (emod_nonneg _ hb)) hbε.le)
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    hb' : LT.lt 0 ↑(abs b)
    hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
    hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
    ⊢ Exists fun t => ∀ (i₀ i₁ : Fin n), Eq (t i₀) (t i₁) → LT.lt (↑(abs (HSub.hSu …
  -/
  refine ⟨fun i ↦ ⟨natAbs (floor ((A i % b : ℤ) / abs b • ε : ℝ)), ?_⟩, ?_⟩
    /-
      case refine_1
      n : Nat
      ε : Real
      hε : LT.lt 0 ε
      b : Int
      hb : Ne b 0
      A : Fin n → Int
      hb' : LT.lt 0 ↑(abs b)
      hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
      hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
      i : Fin n
      ⊢ LT.lt (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) (HSMul.hSMul (abs b) ε))) …
    -/
  · rw [← ofNat_lt, natAbs_of_nonneg (hfloor i), floor_lt, Algebra.smul_def, eq_intCast, ← div_div]
    /-
      case refine_1
      n : Nat
      ε : Real
      hε : LT.lt 0 ε
      b : Int
      hb : Ne b 0
      A : Fin n → Int
      hb' : LT.lt 0 ↑(abs b)
      hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
      hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
      i : Fin n
      ⊢ LT.lt (HDiv.hDiv (HDiv.hDiv ↑(HMod.hMod (A i) b) ↑(abs b)) ε) ↑↑(Nat.ceil (H …
    -/
    apply lt_of_lt_of_le _ (Nat.le_ceil _)
    /-
      n : Nat
      ε : Real
      hε : LT.lt 0 ε
      b : Int
      hb : Ne b 0
      A : Fin n → Int
      hb' : LT.lt 0 ↑(abs b)
      hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
      hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
      i : Fin n
      ⊢ LT.lt (HDiv.hDiv (HDiv.hDiv ↑(HMod.hMod (A i) b) ↑(abs b)) ε) (HDiv.hDiv 1 ε)
    -/
    gcongr
    /-
      case h
      n : Nat
      ε : Real
      hε : LT.lt 0 ε
      b : Int
      hb : Ne b 0
      A : Fin n → Int
      hb' : LT.lt 0 ↑(abs b)
      hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
      hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
      i : Fin n
      ⊢ LT.lt (HDiv.hDiv ↑(HMod.hMod (A i) b) ↑(abs b)) 1
    -/
    rw [div_lt_one hb', cast_lt]
    /-
      case h
      n : Nat
      ε : Real
      hε : LT.lt 0 ε
      b : Int
      hb : Ne b 0
      A : Fin n → Int
      hb' : LT.lt 0 ↑(abs b)
      hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
      hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
      i : Fin n
      ⊢ LT.lt (HMod.hMod (A i) b) (abs b)
    -/
    exact Int.emod_lt _ hb
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    hb' : LT.lt 0 ↑(abs b)
    hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
    hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
    ⊢ ∀ (i₀ i₁ : Fin n), Eq ((fun i => ⟨(Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b …
  -/
  intro i₀ i₁ hi
  have hi : (⌊↑(A i₀ % b) / abs b • ε⌋.natAbs : ℤ) = ⌊↑(A i₁ % b) / abs b • ε⌋.natAbs :=
    congr_arg ((↑) : ℕ → ℤ) (Fin.mk_eq_mk.mp hi)
  /-
    case refine_2
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    hb' : LT.lt 0 ↑(abs b)
    hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
    hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
    i₀ i₁ : Fin n
    hi✝ : Eq ((fun i => ⟨(Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) (HSMul.hSMul …
    hi : Eq ↑(Int.floor (HDiv.hDiv (↑(HMod.hMod (A i₀) b)) (HSMul.hSMul (abs b) ε) …
    ⊢ LT.lt (↑(abs (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)))) (HSMul. …
  -/
  rw [natAbs_of_nonneg (hfloor i₀), natAbs_of_nonneg (hfloor i₁)] at hi
  /-
    case refine_2
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    hb' : LT.lt 0 ↑(abs b)
    hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
    hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
    i₀ i₁ : Fin n
    hi✝ : Eq ((fun i => ⟨(Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) (HSMul.hSMul …
    hi : Eq (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i₀) b)) (HSMul.hSMul (abs b) ε)) …
    ⊢ LT.lt (↑(abs (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)))) (HSMul. …
  -/
  have hi := abs_sub_lt_one_of_floor_eq_floor hi
  /-
    case refine_2
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    hb' : LT.lt 0 ↑(abs b)
    hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
    hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
    i₀ i₁ : Fin n
    hi✝¹ : Eq ((fun i => ⟨(Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) (HSMul.hSMu …
    hi✝ : Eq (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i₀) b)) (HSMul.hSMul (abs b) ε) …
    hi : LT.lt (abs (HSub.hSub (HDiv.hDiv (↑(HMod.hMod (A i₀) b)) (HSMul.hSMul (ab …
    ⊢ LT.lt (↑(abs (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)))) (HSMul. …
  -/
  rw [abs_sub_comm, ← sub_div, abs_div, abs_of_nonneg hbε.le, div_lt_iff₀ hbε, one_mul] at hi
  /-
    case refine_2
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Int
    hb : Ne b 0
    A : Fin n → Int
    hb' : LT.lt 0 ↑(abs b)
    hbε : LT.lt 0 (HSMul.hSMul (abs b) ε)
    hfloor : ∀ (i : Fin n), LE.le 0 (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) ( …
    i₀ i₁ : Fin n
    hi✝¹ : Eq ((fun i => ⟨(Int.floor (HDiv.hDiv (↑(HMod.hMod (A i) b)) (HSMul.hSMu …
    hi✝ : Eq (Int.floor (HDiv.hDiv (↑(HMod.hMod (A i₀) b)) (HSMul.hSMul (abs b) ε) …
    hi : LT.lt (abs (HSub.hSub ↑(HMod.hMod (A i₁) b) ↑(HMod.hMod (A i₀) b))) (HSMu …
    ⊢ LT.lt (↑(abs (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)))) (HSMul. …
  -/
  rwa [Int.cast_abs, Int.cast_sub]
  /-
    🎉 no goals
  -/


/-- `abs : ℤ → ℤ` is an admissible absolute value. -/
noncomputable def absIsAdmissible : IsAdmissible AbsoluteValue.abs :=
  { AbsoluteValue.abs_isEuclidean with
    card := fun ε ↦ ⌈1 / ε⌉₊
    exists_partition' := fun n _ hε _ hb ↦ exists_partition_int n hε hb }


noncomputable instance : Inhabited (IsAdmissible AbsoluteValue.abs) :=
  ⟨absIsAdmissible⟩


