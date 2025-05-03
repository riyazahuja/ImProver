/-- An Akra-Bazzi recurrence is a function that satisfies the recurrence
`T n = (∑ i, a i * T (r i n)) + g n`. -/
structure AkraBazziRecurrence {α : Type*} [Fintype α] [Nonempty α]
    (T : ℕ → ℝ) (g : ℝ → ℝ) (a : α → ℝ) (b : α → ℝ) (r : α → ℕ → ℕ) where
  /-- Point below which the recurrence is in the base case -/
  n₀ : ℕ
  /-- `n₀` is always `> 0` -/
  n₀_gt_zero : 0 < n₀
  /-- The `a`'s are nonzero -/
  a_pos : ∀ i, 0 < a i
  /-- The `b`'s are nonzero -/
  b_pos : ∀ i, 0 < b i
  /-- The b's are less than 1 -/
  b_lt_one : ∀ i, b i < 1
  /-- `g` is nonnegative -/
  g_nonneg : ∀ x ≥ 0, 0 ≤ g x
  /-- `g` grows polynomially -/
  g_grows_poly : AkraBazziRecurrence.GrowsPolynomially g
  /-- The actual recurrence -/
  h_rec (n : ℕ) (hn₀ : n₀ ≤ n) : T n = (∑ i, a i * T (r i n)) + g n
  /-- Base case: `T(n) > 0` whenever `n < n₀` -/
  T_gt_zero' (n : ℕ) (hn : n < n₀) : 0 < T n
  /-- The `r`'s always reduce `n` -/
  r_lt_n : ∀ i n, n₀ ≤ n → r i n < n
  /-- The `r`'s approximate the `b`'s -/
  dist_r_b : ∀ i, (fun n => (r i n : ℝ) - b i * n) =o[atTop] fun n => n / (log n) ^ 2


/-- Smallest `b i` -/
noncomputable def min_bi (b : α → ℝ) : α :=
  Classical.choose <| Finite.exists_min b


/-- Largest `b i` -/
noncomputable def max_bi (b : α → ℝ) : α :=
  Classical.choose <| Finite.exists_max b


@[aesop safe apply]
lemma min_bi_le {b : α → ℝ} (i : α) : b (min_bi b) ≤ b i :=
  Classical.choose_spec (Finite.exists_min b) i


@[aesop safe apply]
lemma max_bi_le {b : α → ℝ} (i : α) : b i ≤ b (max_bi b) :=
  Classical.choose_spec (Finite.exists_max b) i


lemma isLittleO_self_div_log_id :
    (fun (n : ℕ) => n / log n ^ 2) =o[atTop] (fun (n : ℕ) => (n : ℝ)) := by
  calc (fun (n : ℕ) => (n : ℝ) / log n ^ 2) = fun (n : ℕ) => (n : ℝ) * ((log n) ^ 2)⁻¹ := by
                  simp_rw [div_eq_mul_inv]
         _ =o[atTop] fun (n : ℕ) => (n : ℝ) * 1⁻¹ := by
                  refine IsBigO.mul_isLittleO (isBigO_refl _ _) ?_
                  refine IsLittleO.inv_rev ?main ?zero
                  case zero => simp
                  case main => calc
                    _ = (fun (_ : ℕ) => ((1 : ℝ) ^ 2))     := by simp
                    _ =o[atTop] (fun (n : ℕ) => (log n)^2) :=
                          IsLittleO.pow (IsLittleO.natCast_atTop
                            <| isLittleO_const_log_atTop) (by norm_num)
         _ = (fun (n : ℕ) => (n : ℝ)) := by ext; simp


lemma dist_r_b' : ∀ᶠ n in atTop, ∀ i, ‖(r i n : ℝ) - b i * n‖ ≤ n / log n ^ 2 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (Norm.norm (HSub.hSub (↑(r i n) …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (↑(r i x) …
  -/
  intro i
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (↑(r i x)) (HMul.hMu …
  -/
  simpa using IsLittleO.eventuallyLE (R.dist_r_b i)
  /-
    🎉 no goals
  -/


lemma eventually_b_le_r : ∀ᶠ (n : ℕ) in atTop, ∀ i, (b i : ℝ) * n - (n / log n ^ 2) ≤ r i n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HSub.hSub (HMul.hMul (b i) ↑n) …
  -/
  filter_upwards [R.dist_r_b'] with n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : ∀ (i : α), LE.le (Norm.norm (HSub.hSub (↑(r i n)) (HMul.hMul (b i) ↑n)))  …
    ⊢ ∀ (i : α), LE.le (HSub.hSub (HMul.hMul (b i) ↑n) (HDiv.hDiv (↑n) (HPow.hPow  …
  -/
  intro i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : ∀ (i : α), LE.le (Norm.norm (HSub.hSub (↑(r i n)) (HMul.hMul (b i) ↑n)))  …
    i : α
    ⊢ LE.le (HSub.hSub (HMul.hMul (b i) ↑n) (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑ …
  -/
  have h₁ : 0 ≤ b i := le_of_lt <| R.b_pos _
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : ∀ (i : α), LE.le (Norm.norm (HSub.hSub (↑(r i n)) (HMul.hMul (b i) ↑n)))  …
    i : α
    h₁ : LE.le 0 (b i)
    ⊢ LE.le (HSub.hSub (HMul.hMul (b i) ↑n) (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑ …
  -/
  rw [sub_le_iff_le_add, add_comm, ← sub_le_iff_le_add]
  calc (b i : ℝ) * n - r i n = ‖b i * n‖ - ‖(r i n : ℝ)‖ := by
                            simp only [norm_mul, RCLike.norm_natCast, sub_left_inj,
                                       Nat.cast_eq_zero, Real.norm_of_nonneg h₁]
                         _ ≤ ‖(b i * n : ℝ) - r i n‖ := norm_sub_norm_le _ _
                         _ = ‖(r i n : ℝ) - b i * n‖ := norm_sub_rev _ _
                         _ ≤ n / log n ^ 2 := hn i


lemma eventually_r_le_b : ∀ᶠ (n : ℕ) in atTop, ∀ i, r i n ≤ (b i : ℝ) * n + (n / log n ^ 2) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HAdd.hAdd (HMul.hMu …
  -/
  filter_upwards [R.dist_r_b'] with n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : ∀ (i : α), LE.le (Norm.norm (HSub.hSub (↑(r i n)) (HMul.hMul (b i) ↑n)))  …
    ⊢ ∀ (i : α), LE.le (↑(r i n)) (HAdd.hAdd (HMul.hMul (b i) ↑n) (HDiv.hDiv (↑n)  …
  -/
  intro i
  calc r i n = b i * n + (r i n - b i * n) := by ring
             _ ≤ b i * n + ‖r i n - b i * n‖ := by gcongr; exact Real.le_norm_self _
             _ ≤ b i * n + n / log n ^ 2 := by gcongr; exact hn i


lemma eventually_r_lt_n : ∀ᶠ (n : ℕ) in atTop, ∀ i, r i n < n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LT.lt (r i n) n) Filter.atTop
  -/
  filter_upwards [eventually_ge_atTop R.n₀] with n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : LE.le R.n₀ n
    ⊢ ∀ (i : α), LT.lt (r i n) n
  -/
  exact fun i => R.r_lt_n i n hn
  /-
    🎉 no goals
  -/


lemma eventually_bi_mul_le_r : ∀ᶠ (n : ℕ) in atTop, ∀ i, (b (min_bi b) / 2) * n ≤ r i n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraB …
  -/
  have gt_zero : 0 < b (min_bi b) := R.b_pos (min_bi b)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraB …
  -/
  have hlo := isLittleO_self_div_log_id
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    hlo : Asymptotics.IsLittleO Filter.atTop (fun n => HDiv.hDiv (↑n) (HPow.hPow ( …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraB …
  -/
  rw [Asymptotics.isLittleO_iff] at hlo
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraB …
  -/
  have hlo' := hlo (by positivity : 0 < b (min_bi b) / 2)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraB …
  -/
  filter_upwards [hlo', R.eventually_b_le_r] with n hn hn'
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    n : Nat
    hn : LE.le (Norm.norm (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑n) 2))) (HMul.hMul …
    hn' : ∀ (i : α), LE.le (HSub.hSub (HMul.hMul (b i) ↑n) (HDiv.hDiv (↑n) (HPow.h …
    ⊢ ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2) …
  -/
  intro i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    n : Nat
    hn : LE.le (Norm.norm (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑n) 2))) (HMul.hMul …
    hn' : ∀ (i : α), LE.le (HSub.hSub (HMul.hMul (b i) ↑n) (HDiv.hDiv (↑n) (HPow.h …
    i : α
    ⊢ LE.le (HMul.hMul (HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2) ↑n) ↑(r i n)
  -/
  simp only [Real.norm_of_nonneg (by positivity : 0 ≤ (n : ℝ))] at hn
  calc b (min_bi b) / 2 * n = b (min_bi b) * n - b (min_bi b) / 2 * n := by ring
                          _ ≤ b (min_bi b) * n - ‖n / log n ^ 2‖ := by gcongr
                          _ ≤ b i * n - ‖n / log n ^ 2‖ := by gcongr; aesop
                          _ = b i * n - n / log n ^ 2 := by
                                congr
                                exact Real.norm_of_nonneg <| by positivity
                          _ ≤ r i n := hn' i


lemma bi_min_div_two_lt_one : b (min_bi b) / 2 < 1 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ LT.lt (HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2) 1
  -/
  have gt_zero : 0 < b (min_bi b) := R.b_pos (min_bi b)
  calc b (min_bi b) / 2 < b (min_bi b) := by aesop (add safe apply div_two_lt_of_pos)
                      _ < 1 := R.b_lt_one _


                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝¹ : Fintype α
                                                                             T : Nat → Real
                                                                             g : Real → Real
                                                                             a b : α → Real
                                                                             r : α → Nat → Nat
                                                                             inst✝ : Nonempty α
                                                                             R : AkraBazziRecurrence T g a b r
                                                                             ⊢ LT.lt 0 2
                                                                           -/
lemma bi_min_div_two_pos : 0 < b (min_bi b) / 2 := div_pos (R.b_pos _) (by norm_num)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma exists_eventually_const_mul_le_r :
    ∃ c ∈ Set.Ioo (0 : ℝ) 1, ∀ᶠ (n : ℕ) in atTop, ∀ i, c * n ≤ r i n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  have gt_zero : 0 < b (min_bi b) := R.b_pos (min_bi b)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    gt_zero : LT.lt 0 (b (AkraBazziRecurrence.min_bi b))
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  exact ⟨b (min_bi b) / 2, ⟨⟨by positivity, R.bi_min_div_two_lt_one⟩, R.eventually_bi_mul_le_r⟩⟩
  /-
    🎉 no goals
  -/


lemma eventually_r_ge (C : ℝ) : ∀ᶠ (n : ℕ) in atTop, ∀ i, C ≤ r i n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    C : Real
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le C ↑(r i n)) Filter.atTop
  -/
  obtain ⟨c, hc_mem, hc⟩ := R.exists_eventually_const_mul_le_r
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    C c : Real
    hc_mem : Membership.mem (Set.Ioo 0 1) c
    hc : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c ↑n) ↑(r i n)) F …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le C ↑(r i n)) Filter.atTop
  -/
  filter_upwards [eventually_ge_atTop ⌈C / c⌉₊, hc] with n hn₁ hn₂
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    C c : Real
    hc_mem : Membership.mem (Set.Ioo 0 1) c
    hc : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c ↑n) ↑(r i n)) F …
    n : Nat
    hn₁ : LE.le (Nat.ceil (HDiv.hDiv C c)) n
    hn₂ : ∀ (i : α), LE.le (HMul.hMul c ↑n) ↑(r i n)
    ⊢ ∀ (i : α), LE.le C ↑(r i n)
  -/
  have h₁ := hc_mem.1
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    C c : Real
    hc_mem : Membership.mem (Set.Ioo 0 1) c
    hc : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c ↑n) ↑(r i n)) F …
    n : Nat
    hn₁ : LE.le (Nat.ceil (HDiv.hDiv C c)) n
    hn₂ : ∀ (i : α), LE.le (HMul.hMul c ↑n) ↑(r i n)
    h₁ : LT.lt 0 c
    ⊢ ∀ (i : α), LE.le C ↑(r i n)
  -/
  intro i
  calc C = c * (C / c) := by
            rw [← mul_div_assoc]
            exact (mul_div_cancel_left₀ _ (by positivity)).symm
       _ ≤ c * ⌈C / c⌉₊ := by gcongr; simp [Nat.le_ceil]
       _ ≤ c * n := by gcongr
       _ ≤ r i n := hn₂ i


lemma tendsto_atTop_r (i : α) : Tendsto (r i) atTop atTop := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    ⊢ Filter.Tendsto (r i) Filter.atTop Filter.atTop
  -/
  rw [tendsto_atTop]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    ⊢ ∀ (b : Nat), Filter.Eventually (fun a => LE.le b (r i a)) Filter.atTop
  -/
  intro b
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b✝ : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b✝ r
    i : α
    b : Nat
    ⊢ Filter.Eventually (fun a => LE.le b (r i a)) Filter.atTop
  -/
  have := R.eventually_r_ge b
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b✝ : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b✝ r
    i : α
    b : Nat
    this : Filter.Eventually (fun n => ∀ (i : α), LE.le ↑b ↑(r i n)) Filter.atTop
    ⊢ Filter.Eventually (fun a => LE.le b (r i a)) Filter.atTop
  -/
  rw [Filter.eventually_all] at this
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b✝ : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b✝ r
    i : α
    b : Nat
    this : ∀ (i : α), Filter.Eventually (fun x => LE.le ↑b ↑(r i x)) Filter.atTop
    ⊢ Filter.Eventually (fun a => LE.le b (r i a)) Filter.atTop
  -/
  exact_mod_cast this i
  /-
    🎉 no goals
  -/


lemma tendsto_atTop_r_real (i : α) : Tendsto (fun n => (r i n : ℝ)) atTop atTop :=
  Tendsto.comp tendsto_natCast_atTop_atTop (R.tendsto_atTop_r i)


lemma exists_eventually_r_le_const_mul :
    ∃ c ∈ Set.Ioo (0 : ℝ) 1, ∀ᶠ (n : ℕ) in atTop, ∀ i, r i n ≤ c * n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  let c := b (max_bi b) + (1 - b (max_bi b)) / 2
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  have h_max_bi_pos : 0 < b (max_bi b) := R.b_pos _
  have h_max_bi_lt_one : 0 < 1 - b (max_bi b) := by
    have : b (max_bi b) < 1 := R.b_lt_one _
    linarith
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  have hc_pos : 0 < c := by positivity
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  have h₁ : 0 < (1 - b (max_bi b)) / 2 := by positivity
  have hc_lt_one : c < 1 :=
    calc b (max_bi b) + (1 - b (max_bi b)) / 2 = b (max_bi b) * (1 / 2) + 1 / 2 := by ring
                                             _ < 1 * (1 / 2) + 1 / 2 := by
                                                  gcongr
                                                  exact R.b_lt_one _
                                             _ = 1 := by norm_num
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo 0 1) c) (Filter.Eventually (fun …
  -/
  refine ⟨c, ⟨hc_pos, hc_lt_one⟩, ?_⟩
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c ↑n)) Fi …
  -/
  have hlo := isLittleO_self_div_log_id
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    hlo : Asymptotics.IsLittleO Filter.atTop (fun n => HDiv.hDiv (↑n) (HPow.hPow ( …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c ↑n)) Fi …
  -/
  rw [Asymptotics.isLittleO_iff] at hlo
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c ↑n)) Fi …
  -/
  have hlo' := hlo h₁
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c ↑n)) Fi …
  -/
  filter_upwards [hlo', R.eventually_r_le_b] with n hn hn'
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    n : Nat
    hn : LE.le (Norm.norm (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑n) 2))) (HMul.hMul …
    hn' : ∀ (i : α), LE.le (↑(r i n)) (HAdd.hAdd (HMul.hMul (b i) ↑n) (HDiv.hDiv ( …
    ⊢ ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c ↑n)
  -/
  intro i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    n : Nat
    hn : LE.le (Norm.norm (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑n) 2))) (HMul.hMul …
    hn' : ∀ (i : α), LE.le (↑(r i n)) (HAdd.hAdd (HMul.hMul (b i) ↑n) (HDiv.hDiv ( …
    i : α
    ⊢ LE.le (↑(r i n)) (HMul.hMul c ↑n)
  -/
  rw [Real.norm_of_nonneg (by positivity)] at hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c : Real := HAdd.hAdd (b (AkraBazziRecurrence.max_bi b)) (HDiv.hDiv (HSub.hSub …
    h_max_bi_pos : LT.lt 0 (b (AkraBazziRecurrence.max_bi b))
    h_max_bi_lt_one : LT.lt 0 (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b)))
    hc_pos : LT.lt 0 c
    h₁ : LT.lt 0 (HDiv.hDiv (HSub.hSub 1 (b (AkraBazziRecurrence.max_bi b))) 2)
    hc_lt_one : LT.lt c 1
    hlo : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    hlo' : Filter.Eventually (fun x => LE.le (Norm.norm (HDiv.hDiv (↑x) (HPow.hPow …
    n : Nat
    hn : LE.le (HDiv.hDiv (↑n) (HPow.hPow (Real.log ↑n) 2)) (HMul.hMul (HDiv.hDiv  …
    hn' : ∀ (i : α), LE.le (↑(r i n)) (HAdd.hAdd (HMul.hMul (b i) ↑n) (HDiv.hDiv ( …
    i : α
    ⊢ LE.le (↑(r i n)) (HMul.hMul c ↑n)
  -/
  simp only [Real.norm_of_nonneg (by positivity : 0 ≤ (n : ℝ))] at hn
  calc r i n ≤ b i * n + n / log n ^ 2 := by exact hn' i
             _ ≤ b i * n + (1 - b (max_bi b)) / 2 * n := by gcongr
             _ = (b i + (1 - b (max_bi b)) / 2) * n := by ring
             _ ≤ (b (max_bi b) + (1 - b (max_bi b)) / 2) * n := by gcongr; exact max_bi_le _


lemma eventually_r_pos : ∀ᶠ (n : ℕ) in atTop, ∀ i, 0 < r i n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LT.lt 0 (r i n)) Filter.atTop
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LT.lt 0 (r i x)) Filter.atTop
  -/
  exact fun i => (R.tendsto_atTop_r i).eventually_gt_atTop 0
  /-
    🎉 no goals
  -/


lemma eventually_log_b_mul_pos : ∀ᶠ (n : ℕ) in atTop, ∀ i, 0 < log (b i * n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LT.lt 0 (Real.log (HMul.hMul (b i) ↑n …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LT.lt 0 (Real.log (HMul.hMul (b i) ↑x …
  -/
  intro i
  have h : Tendsto (fun (n : ℕ) => log (b i * n)) atTop atTop :=
    Tendsto.comp tendsto_log_atTop
      <| Tendsto.const_mul_atTop (b_pos R i) tendsto_natCast_atTop_atTop
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    h : Filter.Tendsto (fun n => Real.log (HMul.hMul (b i) ↑n)) Filter.atTop Filte …
    ⊢ Filter.Eventually (fun x => LT.lt 0 (Real.log (HMul.hMul (b i) ↑x))) Filter. …
  -/
  exact h.eventually_gt_atTop 0
  /-
    🎉 no goals
  -/


@[aesop safe apply] lemma T_pos (n : ℕ) : 0 < T n := by
  induction n using Nat.strongRecOn with
  | ind n h_ind =>
    cases lt_or_le n R.n₀ with
    | inl hn => exact R.T_gt_zero' n hn -- n < R.n₀
    | inr hn => -- R.n₀ ≤ n
      rw [R.h_rec n hn]
      have := R.g_nonneg
      refine add_pos_of_pos_of_nonneg (Finset.sum_pos ?sum_elems univ_nonempty) (by aesop)
      exact fun i _ => mul_pos (R.a_pos i) <| h_ind _ (R.r_lt_n i _ hn)


@[aesop safe apply]
lemma T_nonneg (n : ℕ) : 0 ≤ T n := le_of_lt <| R.T_pos n


/-- The "smoothing function" is defined as `1 / log n`. This is defined as an `ℝ → ℝ` function
as opposed to `ℕ → ℝ` since this is more convenient for the proof, where we need to e.g. take
derivatives. -/
noncomputable def smoothingFn (n : ℝ) : ℝ := 1 / log n


local notation "ε" => smoothingFn


lemma one_add_smoothingFn_le_two {x : ℝ} (hx : exp 1 ≤ x) : 1 + ε x ≤ 2 := by
  /-
    x : Real
    hx : LE.le (Real.exp 1) x
    ⊢ LE.le (HAdd.hAdd 1 (AkraBazziRecurrence.smoothingFn x)) 2
  -/
  simp only [smoothingFn, ← one_add_one_eq_two]
  /-
    x : Real
    hx : LE.le (Real.exp 1) x
    ⊢ LE.le (HAdd.hAdd 1 (HDiv.hDiv 1 (Real.log x))) (HAdd.hAdd 1 1)
  -/
  gcongr
  have : 1 < x := by
    calc 1 = exp 0 := by simp
         _ < exp 1 := by simp
         _ ≤ x := hx
  /-
    case bc
    x : Real
    hx : LE.le (Real.exp 1) x
    this : LT.lt 1 x
    ⊢ LE.le (HDiv.hDiv 1 (Real.log x)) 1
  -/
  rw [div_le_one (log_pos this)]
  calc 1 = log (exp 1) := by simp
       _ ≤ log x := log_le_log (exp_pos _) hx


lemma isLittleO_smoothingFn_one : ε =o[atTop] (fun _ => (1 : ℝ)) := by
  /-
    ⊢ Asymptotics.IsLittleO Filter.atTop AkraBazziRecurrence.smoothingFn fun x => 1
  -/
  unfold smoothingFn
  /-
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HDiv.hDiv 1 (Real.log n)) fun x …
  -/
  refine isLittleO_of_tendsto (fun _ h => False.elim <| one_ne_zero h) ?_
  /-
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HDiv.hDiv 1 (Real.log x)) 1) Filter.atTo …
  -/
  simp only [one_div, div_one]
  /-
    ⊢ Filter.Tendsto (fun x => Inv.inv (Real.log x)) Filter.atTop (nhds 0)
  -/
  exact Tendsto.inv_tendsto_atTop Real.tendsto_log_atTop
  /-
    🎉 no goals
  -/


lemma isEquivalent_one_add_smoothingFn_one : (fun x => 1 + ε x) ~[atTop] (fun _ => (1 : ℝ)) :=
  IsEquivalent.add_isLittleO IsEquivalent.refl isLittleO_smoothingFn_one


lemma isEquivalent_one_sub_smoothingFn_one : (fun x => 1 - ε x) ~[atTop] (fun _ => (1 : ℝ)) :=
  IsEquivalent.sub_isLittleO IsEquivalent.refl isLittleO_smoothingFn_one


lemma growsPolynomially_one_sub_smoothingFn : GrowsPolynomially fun x => 1 - ε x :=
  GrowsPolynomially.of_isEquivalent_const isEquivalent_one_sub_smoothingFn_one


lemma growsPolynomially_one_add_smoothingFn : GrowsPolynomially fun x => 1 + ε x :=
  GrowsPolynomially.of_isEquivalent_const isEquivalent_one_add_smoothingFn_one


lemma eventually_one_sub_smoothingFn_gt_const_real (c : ℝ) (hc : c < 1) :
    ∀ᶠ (x : ℝ) in atTop, c < 1 - ε x := by
  have h₁ : Tendsto (fun x => 1 - ε x) atTop (𝓝 1) := by
    rw [← isEquivalent_const_iff_tendsto one_ne_zero]
    exact isEquivalent_one_sub_smoothingFn_one
  /-
    c : Real
    hc : LT.lt c 1
    h₁ : Filter.Tendsto (fun x => HSub.hSub 1 (AkraBazziRecurrence.smoothingFn x)) …
    ⊢ Filter.Eventually (fun x => LT.lt c (HSub.hSub 1 (AkraBazziRecurrence.smooth …
  -/
  rw [tendsto_order] at h₁
  /-
    c : Real
    hc : LT.lt c 1
    h₁ : And (∀ (a' : Real), LT.lt a' 1 → Filter.Eventually (fun b => LT.lt a' (HS …
    ⊢ Filter.Eventually (fun x => LT.lt c (HSub.hSub 1 (AkraBazziRecurrence.smooth …
  -/
  exact h₁.1 c hc
  /-
    🎉 no goals
  -/


lemma eventually_one_sub_smoothingFn_gt_const (c : ℝ) (hc : c < 1) :
    ∀ᶠ (n : ℕ) in atTop, c < 1 - ε n :=
  Eventually.natCast_atTop (p := fun n => c < 1 - ε n)
    <| eventually_one_sub_smoothingFn_gt_const_real c hc


lemma eventually_one_sub_smoothingFn_pos_real : ∀ᶠ (x : ℝ) in atTop, 0 < 1 - ε x :=
  eventually_one_sub_smoothingFn_gt_const_real 0 zero_lt_one


lemma eventually_one_sub_smoothingFn_pos : ∀ᶠ (n : ℕ) in atTop, 0 < 1 - ε n :=
  (eventually_one_sub_smoothingFn_pos_real).natCast_atTop


lemma eventually_one_sub_smoothingFn_nonneg : ∀ᶠ (n : ℕ) in atTop, 0 ≤ 1 - ε n := by
  /-
    ⊢ Filter.Eventually (fun n => LE.le 0 (HSub.hSub 1 (AkraBazziRecurrence.smooth …
  -/
  filter_upwards [eventually_one_sub_smoothingFn_pos] with n hn; exact le_of_lt hn
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


include R in
lemma eventually_one_sub_smoothingFn_r_pos : ∀ᶠ (n : ℕ) in atTop, ∀ i, 0 < 1 - ε (r i n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LT.lt 0 (HSub.hSub 1 (AkraBazziRecurr …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LT.lt 0 (HSub.hSub 1 (AkraBazziRecurr …
  -/
  exact fun i => (R.tendsto_atTop_r_real i).eventually eventually_one_sub_smoothingFn_pos_real
  /-
    🎉 no goals
  -/


@[aesop safe apply]
lemma differentiableAt_smoothingFn {x : ℝ} (hx : 1 < x) : DifferentiableAt ℝ ε x := by
  /-
    x : Real
    hx : LT.lt 1 x
    ⊢ DifferentiableAt Real AkraBazziRecurrence.smoothingFn x
  -/
  have : log x ≠ 0 := Real.log_ne_zero_of_pos_of_ne_one (by positivity) (ne_of_gt hx)
  /-
    x : Real
    hx : LT.lt 1 x
    this : Ne (Real.log x) 0
    ⊢ DifferentiableAt Real AkraBazziRecurrence.smoothingFn x
  -/
  show DifferentiableAt ℝ (fun z => 1 / log z) x
  /-
    x : Real
    hx : LT.lt 1 x
    this : Ne (Real.log x) 0
    ⊢ DifferentiableAt Real (fun z => HDiv.hDiv 1 (Real.log z)) x
  -/
  simp_rw [one_div]
  /-
    x : Real
    hx : LT.lt 1 x
    this : Ne (Real.log x) 0
    ⊢ DifferentiableAt Real (fun z => Inv.inv (Real.log z)) x
  -/
  exact DifferentiableAt.inv (differentiableAt_log (by positivity)) this
  /-
    🎉 no goals
  -/


@[aesop safe apply]
lemma differentiableAt_one_sub_smoothingFn {x : ℝ} (hx : 1 < x) :
    DifferentiableAt ℝ (fun z => 1 - ε z) x :=
  DifferentiableAt.sub (differentiableAt_const _) <| differentiableAt_smoothingFn hx


lemma differentiableOn_one_sub_smoothingFn : DifferentiableOn ℝ (fun z => 1 - ε z) (Set.Ioi 1) :=
  fun _ hx => (differentiableAt_one_sub_smoothingFn hx).differentiableWithinAt


@[aesop safe apply]
lemma differentiableAt_one_add_smoothingFn {x : ℝ} (hx : 1 < x) :
    DifferentiableAt ℝ (fun z => 1 + ε z) x :=
  DifferentiableAt.add (differentiableAt_const _) <| differentiableAt_smoothingFn hx


lemma differentiableOn_one_add_smoothingFn : DifferentiableOn ℝ (fun z => 1 + ε z) (Set.Ioi 1) :=
  fun _ hx => (differentiableAt_one_add_smoothingFn hx).differentiableWithinAt


lemma deriv_smoothingFn {x : ℝ} (hx : 1 < x) : deriv ε x = -x⁻¹ / (log x ^ 2) := by
  /-
    x : Real
    hx : LT.lt 1 x
    ⊢ Eq (deriv AkraBazziRecurrence.smoothingFn x) (HDiv.hDiv (Neg.neg (Inv.inv x) …
  -/
  have : log x ≠ 0 := Real.log_ne_zero_of_pos_of_ne_one (by positivity) (ne_of_gt hx)
  /-
    x : Real
    hx : LT.lt 1 x
    this : Ne (Real.log x) 0
    ⊢ Eq (deriv AkraBazziRecurrence.smoothingFn x) (HDiv.hDiv (Neg.neg (Inv.inv x) …
  -/
  show deriv (fun z => 1 / log z) x = -x⁻¹ / (log x ^ 2)
  /-
    x : Real
    hx : LT.lt 1 x
    this : Ne (Real.log x) 0
    ⊢ Eq (deriv (fun z => HDiv.hDiv 1 (Real.log z)) x) (HDiv.hDiv (Neg.neg (Inv.in …
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
  rw [deriv_div] <;> aesop
                     /-
                       🎉 no goals
                     -/


lemma isLittleO_deriv_smoothingFn : deriv ε =o[atTop] fun x => x⁻¹ := calc
  deriv ε =ᶠ[atTop] fun x => -x⁻¹ / (log x ^ 2) := by
            /-
              ⊢ Filter.atTop.EventuallyEq (deriv AkraBazziRecurrence.smoothingFn) fun x => H …
            -/
            filter_upwards [eventually_gt_atTop 1] with x hx
            /-
              case h
              x : Real
              hx : LT.lt 1 x
              ⊢ Eq (deriv AkraBazziRecurrence.smoothingFn x) (HDiv.hDiv (Neg.neg (Inv.inv x) …
            -/
            rw [deriv_smoothingFn hx]
            /-
              🎉 no goals
            -/
    _ = fun x => (-x * log x ^ 2)⁻¹ := by
            /-
              ⊢ Eq (fun x => HDiv.hDiv (Neg.neg (Inv.inv x)) (HPow.hPow (Real.log x) 2)) fun …
            -/
            simp_rw [neg_div, div_eq_mul_inv, ← mul_inv, neg_inv, neg_mul]
            /-
              🎉 no goals
            -/
    _ =o[atTop] fun x => (x * 1)⁻¹ := by
            /-
              ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Inv.inv (HMul.hMul (Neg.neg x)  …
            -/
            refine IsLittleO.inv_rev ?_ ?_
            · refine IsBigO.mul_isLittleO
                (by rw [isBigO_neg_right]; aesop (add safe isBigO_refl)) ?_
              /-
                case refine_1
                ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => 1) fun x => HPow.hPow (Real.log …
              -/
              rw [isLittleO_one_left_iff]
              exact Tendsto.comp tendsto_norm_atTop_atTop
                <| Tendsto.comp (tendsto_pow_atTop (by norm_num)) tendsto_log_atTop
              /-
                case refine_2
                ⊢ Filter.Eventually (fun x => Eq (HMul.hMul x 1) 0 → Eq (HMul.hMul (Neg.neg x) …
              -/
            · exact Filter.Eventually.of_forall (fun x hx => by rw [mul_one] at hx; simp [hx])
              /-
                🎉 no goals
              -/
                           /-
                             ⊢ Eq (fun x => Inv.inv (HMul.hMul x 1)) fun x => Inv.inv x
                           -/
    _ = fun x => x⁻¹ := by simp
                           /-
                             🎉 no goals
                           -/


lemma eventually_deriv_one_sub_smoothingFn :
    deriv (fun x => 1 - ε x) =ᶠ[atTop] fun x => x⁻¹ / (log x ^ 2) := calc
  deriv (fun x => 1 - ε x) =ᶠ[atTop] -(deriv ε) := by
        /-
          ⊢ Filter.atTop.EventuallyEq (deriv fun x => HSub.hSub 1 (AkraBazziRecurrence.s …
        -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
        filter_upwards [eventually_gt_atTop 1] with x hx; rw [deriv_sub] <;> aesop
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    _ =ᶠ[atTop] fun x => x⁻¹ / (log x ^ 2) := by
        /-
          ⊢ Filter.atTop.EventuallyEq (Neg.neg (deriv AkraBazziRecurrence.smoothingFn))  …
        -/
        filter_upwards [eventually_gt_atTop 1] with x hx
        /-
          case h
          x : Real
          hx : LT.lt 1 x
          ⊢ Eq (Neg.neg (deriv AkraBazziRecurrence.smoothingFn) x) (HDiv.hDiv (Inv.inv x …
        -/
        simp [deriv_smoothingFn hx, neg_div]
        /-
          🎉 no goals
        -/


lemma eventually_deriv_one_add_smoothingFn :
    deriv (fun x => 1 + ε x) =ᶠ[atTop] fun x => -x⁻¹ / (log x ^ 2) := calc
  deriv (fun x => 1 + ε x) =ᶠ[atTop] deriv ε := by
          /-
            ⊢ Filter.atTop.EventuallyEq (deriv fun x => HAdd.hAdd 1 (AkraBazziRecurrence.s …
          -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          filter_upwards [eventually_gt_atTop 1] with x hx; rw [deriv_add] <;> aesop
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    _ =ᶠ[atTop] fun x => -x⁻¹ / (log x ^ 2) := by
          /-
            ⊢ Filter.atTop.EventuallyEq (deriv AkraBazziRecurrence.smoothingFn) fun x => H …
          -/
          filter_upwards [eventually_gt_atTop 1] with x hx
          /-
            case h
            x : Real
            hx : LT.lt 1 x
            ⊢ Eq (deriv AkraBazziRecurrence.smoothingFn x) (HDiv.hDiv (Neg.neg (Inv.inv x) …
          -/
          simp [deriv_smoothingFn hx]
          /-
            🎉 no goals
          -/


lemma isLittleO_deriv_one_sub_smoothingFn :
    deriv (fun x => 1 - ε x) =o[atTop] fun (x : ℝ) => x⁻¹ := calc
  deriv (fun x => 1 - ε x) =ᶠ[atTop] fun z => -(deriv ε z) := by
          /-
            ⊢ Filter.atTop.EventuallyEq (deriv fun x => HSub.hSub 1 (AkraBazziRecurrence.s …
          -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          filter_upwards [eventually_gt_atTop 1] with x hx; rw [deriv_sub] <;> aesop
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                   /-
                                     ⊢ Asymptotics.IsLittleO Filter.atTop (fun z => Neg.neg (deriv AkraBazziRecurre …
                                   -/
    _ =o[atTop] fun x => x⁻¹ := by rw [isLittleO_neg_left]; exact isLittleO_deriv_smoothingFn
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma isLittleO_deriv_one_add_smoothingFn :
    deriv (fun x => 1 + ε x) =o[atTop] fun (x : ℝ) => x⁻¹ := calc
  deriv (fun x => 1 + ε x) =ᶠ[atTop] fun z => deriv ε z := by
          /-
            ⊢ Filter.atTop.EventuallyEq (deriv fun x => HAdd.hAdd 1 (AkraBazziRecurrence.s …
          -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          filter_upwards [eventually_gt_atTop 1] with x hx; rw [deriv_add] <;> aesop
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    _ =o[atTop] fun x => x⁻¹ := isLittleO_deriv_smoothingFn


lemma eventually_one_add_smoothingFn_pos : ∀ᶠ (n : ℕ) in atTop, 0 < 1 + ε n := by
  /-
    ⊢ Filter.Eventually (fun n => LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smooth …
  -/
  have h₁ := isLittleO_smoothingFn_one
  /-
    h₁ : Asymptotics.IsLittleO Filter.atTop AkraBazziRecurrence.smoothingFn fun x  …
    ⊢ Filter.Eventually (fun n => LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smooth …
  -/
  rw [isLittleO_iff] at h₁
  /-
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (A …
    ⊢ Filter.Eventually (fun n => LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smooth …
  -/
  refine Eventually.natCast_atTop (p := fun n => 0 < 1 + ε n) ?_
  /-
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (A …
    ⊢ Filter.Eventually (fun x => (fun n => LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurre …
  -/
  filter_upwards [h₁ (by norm_num : (0 : ℝ) < 1/2), eventually_gt_atTop 1] with x _ hx'
  /-
    case h
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (A …
    x : Real
    a✝ : LE.le (Norm.norm (AkraBazziRecurrence.smoothingFn x)) (HMul.hMul (1 / 2)  …
    hx' : LT.lt 1 x
    ⊢ LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smoothingFn x))
  -/
  have : 0 < log x := Real.log_pos hx'
  /-
    case h
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (A …
    x : Real
    a✝ : LE.le (Norm.norm (AkraBazziRecurrence.smoothingFn x)) (HMul.hMul (1 / 2)  …
    hx' : LT.lt 1 x
    this : LT.lt 0 (Real.log x)
    ⊢ LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smoothingFn x))
  -/
  show 0 < 1 + 1 / log x
  /-
    case h
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (A …
    x : Real
    a✝ : LE.le (Norm.norm (AkraBazziRecurrence.smoothingFn x)) (HMul.hMul (1 / 2)  …
    hx' : LT.lt 1 x
    this : LT.lt 0 (Real.log x)
    ⊢ LT.lt 0 (HAdd.hAdd 1 (HDiv.hDiv 1 (Real.log x)))
  -/
  positivity
  /-
    🎉 no goals
  -/


include R in
lemma eventually_one_add_smoothingFn_r_pos : ∀ᶠ (n : ℕ) in atTop, ∀ i, 0 < 1 + ε (r i n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurr …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurr …
  -/
  exact fun i => (R.tendsto_atTop_r i).eventually (f := r i) eventually_one_add_smoothingFn_pos
  /-
    🎉 no goals
  -/


lemma eventually_one_add_smoothingFn_nonneg : ∀ᶠ (n : ℕ) in atTop, 0 ≤ 1 + ε n := by
  /-
    ⊢ Filter.Eventually (fun n => LE.le 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smooth …
  -/
  filter_upwards [eventually_one_add_smoothingFn_pos] with n hn; exact le_of_lt hn
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma strictAntiOn_smoothingFn : StrictAntiOn ε (Set.Ioi 1) := by
  /-
    ⊢ StrictAntiOn AkraBazziRecurrence.smoothingFn (Set.Ioi 1)
  -/
  show StrictAntiOn (fun x => 1 / log x) (Set.Ioi 1)
  /-
    ⊢ StrictAntiOn (fun x => HDiv.hDiv 1 (Real.log x)) (Set.Ioi 1)
  -/
  simp_rw [one_div]
  /-
    ⊢ StrictAntiOn (fun x => Inv.inv (Real.log x)) (Set.Ioi 1)
  -/
  refine StrictAntiOn.comp_strictMonoOn inv_strictAntiOn ?log fun _ hx => log_pos hx
  /-
    case log
    ⊢ StrictMonoOn Real.log (Set.Ioi 1)
  -/
  refine StrictMonoOn.mono strictMonoOn_log (fun x hx => ?_)
  /-
    case log
    x : Real
    hx : Membership.mem (Set.Ioi 1) x
    ⊢ Membership.mem (Set.Ioi 0) x
  -/
  exact Set.Ioi_subset_Ioi zero_le_one hx
  /-
    🎉 no goals
  -/


lemma strictMonoOn_one_sub_smoothingFn :
    StrictMonoOn (fun (x : ℝ) => (1 : ℝ) - ε x) (Set.Ioi 1) := by
  /-
    ⊢ StrictMonoOn (fun x => HSub.hSub 1 (AkraBazziRecurrence.smoothingFn x)) (Set …
  -/
  simp_rw [sub_eq_add_neg]
  /-
    ⊢ StrictMonoOn (fun x => HAdd.hAdd 1 (Neg.neg (AkraBazziRecurrence.smoothingFn …
  -/
  exact StrictMonoOn.const_add (StrictAntiOn.neg <| strictAntiOn_smoothingFn) 1
  /-
    🎉 no goals
  -/


lemma strictAntiOn_one_add_smoothingFn : StrictAntiOn (fun (x : ℝ) => (1 : ℝ) + ε x) (Set.Ioi 1) :=
  StrictAntiOn.const_add strictAntiOn_smoothingFn 1


lemma isEquivalent_smoothingFn_sub_self (i : α) :
    (fun (n : ℕ) => ε (b i * n) - ε n) ~[atTop] fun n => -log (b i) / (log n)^2 := by
  calc (fun (n : ℕ) => 1 / log (b i * n) - 1 / log n)
        =ᶠ[atTop] fun (n : ℕ) => (log n - log (b i * n)) / (log (b i * n) * log n) := by
            filter_upwards [eventually_gt_atTop 1, R.eventually_log_b_mul_pos] with n hn hn'
            have h_log_pos : 0 < log n := Real.log_pos <| by aesop
            simp only [one_div]
            rw [inv_sub_inv (by have := hn' i; positivity) (by aesop)]
      _ =ᶠ[atTop] (fun (n : ℕ) ↦ (log n - log (b i) - log n) / ((log (b i) + log n) * log n)) := by
            filter_upwards [eventually_ne_atTop 0] with n hn
            have : 0 < b i := R.b_pos i
            rw [log_mul (by positivity) (by aesop), sub_add_eq_sub_sub]
      _ = (fun (n : ℕ) => -log (b i) / ((log (b i) + log n) * log n)) := by ext; congr; ring
      _ ~[atTop] (fun (n : ℕ) => -log (b i) / (log n * log n)) := by
            refine IsEquivalent.div (IsEquivalent.refl) <| IsEquivalent.mul ?_ (IsEquivalent.refl)
            have : (fun (n : ℕ) => log (b i) + log n) = fun (n : ℕ) => log n + log (b i) := by
              ext; simp [add_comm]
            rw [this]
            exact IsEquivalent.add_isLittleO IsEquivalent.refl
              <| IsLittleO.natCast_atTop (f := fun (_ : ℝ) => log (b i))
                isLittleO_const_log_atTop
      _ = (fun (n : ℕ) => -log (b i) / (log n)^2) := by ext; congr 1; rw [← pow_two]


lemma isTheta_smoothingFn_sub_self (i : α) :
    (fun (n : ℕ) => ε (b i * n) - ε n) =Θ[atTop] fun n => 1 / (log n)^2 := by
  calc (fun (n : ℕ) => ε (b i * n) - ε n) =Θ[atTop] fun n => (-log (b i)) / (log n)^2 := by
                  exact (R.isEquivalent_smoothingFn_sub_self i).isTheta
    _ = fun (n : ℕ) => (-log (b i)) * 1 / (log n)^2 := by simp only [mul_one]
    _ = fun (n : ℕ) => -log (b i) * (1 / (log n)^2) := by simp_rw [← mul_div_assoc]
    _ =Θ[atTop] fun (n : ℕ) => 1 / (log n)^2 := by
                  have : -log (b i) ≠ 0 := by
                    rw [neg_ne_zero]
                    exact Real.log_ne_zero_of_pos_of_ne_one
                            (R.b_pos i) (ne_of_lt <| R.b_lt_one i)
                  rw [← isTheta_const_mul_right this]



@[continuity]
lemma continuous_sumCoeffsExp : Continuous (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Continuous fun p => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow (b i …
  -/
  refine continuous_finset_sum Finset.univ fun i _ => Continuous.mul (by continuity) ?_
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    x✝ : Membership.mem Finset.univ i
    ⊢ Continuous (HPow.hPow (b i))
  -/
  exact Continuous.rpow continuous_const continuous_id (fun x => Or.inl (ne_of_gt (R.b_pos i)))
  /-
    🎉 no goals
  -/


lemma strictAnti_sumCoeffsExp : StrictAnti (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ StrictAnti fun p => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow (b i …
  -/
  rw [← Finset.sum_fn]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ StrictAnti (Finset.univ.sum fun c p => HMul.hMul (a c) (HPow.hPow (b c) p))
  -/
  refine Finset.sum_induction_nonempty _ _ (fun _ _ => StrictAnti.add) univ_nonempty ?terms
  /-
    case terms
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (x : α), Membership.mem Finset.univ x → StrictAnti fun p => HMul.hMul (a x …
  -/
  refine fun i _ => StrictAnti.const_mul ?_ (R.a_pos i)
  /-
    case terms
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    x✝ : Membership.mem Finset.univ i
    ⊢ StrictAnti (HPow.hPow (b i))
  -/
  exact Real.strictAnti_rpow_of_base_lt_one (R.b_pos i) (R.b_lt_one i)
  /-
    🎉 no goals
  -/


lemma tendsto_zero_sumCoeffsExp : Tendsto (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) atTop (𝓝 0) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Tendsto (fun p => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow …
  -/
  have h₁ : Finset.univ.sum (fun _ : α => (0 : ℝ)) = 0 := by simp
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Eq (Finset.univ.sum fun x => 0) 0
    ⊢ Filter.Tendsto (fun p => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow …
  -/
  rw [← h₁]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Eq (Finset.univ.sum fun x => 0) 0
    ⊢ Filter.Tendsto (fun p => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow …
  -/
  refine tendsto_finset_sum (univ : Finset α) (fun i _ => ?_)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Eq (Finset.univ.sum fun x => 0) 0
    i : α
    x✝ : Membership.mem Finset.univ i
    ⊢ Filter.Tendsto (fun p => HMul.hMul (a i) (HPow.hPow (b i) p)) Filter.atTop ( …
  -/
  rw [← mul_zero (a i)]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Eq (Finset.univ.sum fun x => 0) 0
    i : α
    x✝ : Membership.mem Finset.univ i
    ⊢ Filter.Tendsto (fun p => HMul.hMul (a i) (HPow.hPow (b i) p)) Filter.atTop ( …
  -/
  refine Tendsto.mul (by simp) <| tendsto_rpow_atTop_of_base_lt_one _ ?_ (R.b_lt_one i)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Eq (Finset.univ.sum fun x => 0) 0
    i : α
    x✝ : Membership.mem Finset.univ i
    ⊢ LT.lt (-1) (b i)
  -/
  have := R.b_pos i
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Eq (Finset.univ.sum fun x => 0) 0
    i : α
    x✝ : Membership.mem Finset.univ i
    this : LT.lt 0 (b i)
    ⊢ LT.lt (-1) (b i)
  -/
  linarith
  /-
    🎉 no goals
  -/


lemma tendsto_atTop_sumCoeffsExp : Tendsto (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) atBot atTop := by
  have h₁ : Tendsto (fun p : ℝ => (a (max_bi b) : ℝ) * b (max_bi b) ^ p) atBot atTop :=
    Tendsto.mul_atTop (R.a_pos (max_bi b)) (by simp)
      <| tendsto_rpow_atBot_of_base_lt_one _
      (by have := R.b_pos (max_bi b); linarith) (R.b_lt_one _)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Filter.Tendsto (fun p => HMul.hMul (a (AkraBazziRecurrence.max_bi b)) (HP …
    ⊢ Filter.Tendsto (fun p => Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow …
  -/
  refine tendsto_atTop_mono (fun p => ?_) h₁
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Filter.Tendsto (fun p => HMul.hMul (a (AkraBazziRecurrence.max_bi b)) (HP …
    p : Real
    ⊢ LE.le (HMul.hMul (a (AkraBazziRecurrence.max_bi b)) (HPow.hPow (b (AkraBazzi …
  -/
  refine Finset.single_le_sum (f := fun i => (a i : ℝ) * b i ^ p) (fun i _ => ?_) (mem_univ _)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁ : Filter.Tendsto (fun p => HMul.hMul (a (AkraBazziRecurrence.max_bi b)) (HP …
    p : Real
    i : α
    x✝ : Membership.mem Finset.univ i
    ⊢ LE.le 0 ((fun i => HMul.hMul (a i) (HPow.hPow (b i) p)) i)
  -/
  have h₁ : 0 < a i := R.a_pos i
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁✝ : Filter.Tendsto (fun p => HMul.hMul (a (AkraBazziRecurrence.max_bi b)) (H …
    p : Real
    i : α
    x✝ : Membership.mem Finset.univ i
    h₁ : LT.lt 0 (a i)
    ⊢ LE.le 0 ((fun i => HMul.hMul (a i) (HPow.hPow (b i) p)) i)
  -/
  have h₂ : 0 < b i := R.b_pos i
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    h₁✝ : Filter.Tendsto (fun p => HMul.hMul (a (AkraBazziRecurrence.max_bi b)) (H …
    p : Real
    i : α
    x✝ : Membership.mem Finset.univ i
    h₁ : LT.lt 0 (a i)
    h₂ : LT.lt 0 (b i)
    ⊢ LE.le 0 ((fun i => HMul.hMul (a i) (HPow.hPow (b i) p)) i)
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma one_mem_range_sumCoeffsExp : 1 ∈ Set.range (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Membership.mem (Set.range fun p => Finset.univ.sum fun i => HMul.hMul (a i)  …
  -/
  refine mem_range_of_exists_le_of_exists_ge R.continuous_sumCoeffsExp ?le_one ?ge_one
  case le_one =>
    exact R.tendsto_zero_sumCoeffsExp.eventually_le_const zero_lt_one |>.exists
  case ge_one =>
    exact R.tendsto_atTop_sumCoeffsExp.eventually_ge_atTop _ |>.exists


/-- The function x ↦ ∑ a_i b_i^x is injective. This implies the uniqueness of `p`. -/
lemma injective_sumCoeffsExp : Function.Injective (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) :=
    R.strictAnti_sumCoeffsExp.injective


variable (a b) in
/-- The exponent `p` associated with a particular Akra-Bazzi recurrence. -/
noncomputable irreducible_def p : ℝ := Function.invFun (fun (p : ℝ) => ∑ i, a i * (b i) ^ p) 1


include R in
@[simp]
lemma sumCoeffsExp_p_eq_one : ∑ i, a i * (b i) ^ p a b = 1 := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (a i) (HPow.hPow (b i) (AkraBazziRecu …
  -/
  simp only [p]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (a x) (HPow.hPow (b x) (Function.invF …
  -/
  exact Function.invFun_eq (by rw [← Set.mem_range]; exact R.one_mem_range_sumCoeffsExp)
  /-
    🎉 no goals
  -/


/-- The transformation which turns a function `g` into
`n^p * ∑ u ∈ Finset.Ico n₀ n, g u / u^(p+1)`. -/
noncomputable def sumTransform (p : ℝ) (g : ℝ → ℝ) (n₀ n : ℕ) :=
  n^p * ∑ u ∈ Finset.Ico n₀ n, g u / u^(p + 1)


lemma sumTransform_def {p : ℝ} {g : ℝ → ℝ} {n₀ n : ℕ} :
    sumTransform p g n₀ n = n^p * ∑ u ∈ Finset.Ico n₀ n, g u / u^(p + 1) := rfl



/-- The asymptotic bound satisfied by an Akra-Bazzi recurrence, namely
`n^p (1 + ∑_{u < n} g(u) / u^(p+1))`. -/
noncomputable def asympBound (n : ℕ) : ℝ := n ^ p a b + sumTransform (p a b) g 0 n


lemma asympBound_def {α} [Fintype α] (a b : α → ℝ) {n : ℕ} :
    asympBound g a b n = n ^ p a b + sumTransform (p a b) g 0 n := rfl


lemma asympBound_def' {α} [Fintype α] (a b : α → ℝ) {n : ℕ} :
    asympBound g a b n = n ^ p a b * (1 + (∑ u ∈ range n, g u / u ^ (p a b + 1))) := by
  /-
    g : Real → Real
    α : Type u_2
    inst✝ : Fintype α
    a b : α → Real
    n : Nat
    ⊢ Eq (AkraBazziRecurrence.asympBound g a b n) (HMul.hMul (HPow.hPow (↑n) (Akra …
  -/
  simp [asympBound_def, sumTransform, mul_add, mul_one, Finset.sum_Ico_eq_sum_range]
  /-
    🎉 no goals
  -/


lemma asympBound_pos (n : ℕ) (hn : 0 < n) : 0 < asympBound g a b n := by
  calc 0 < (n : ℝ) ^ p a b * (1 + 0) := by aesop (add safe Real.rpow_pos_of_pos)
       _ ≤ asympBound g a b n := by
                    simp only [asympBound_def']
                    gcongr n^p a b * (1 + ?_)
                    have := R.g_nonneg
                    aesop (add safe Real.rpow_nonneg,
                               safe div_nonneg,
                               safe Finset.sum_nonneg)


lemma eventually_asympBound_pos : ∀ᶠ (n : ℕ) in atTop, 0 < asympBound g a b n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => LT.lt 0 (AkraBazziRecurrence.asympBound g a b n) …
  -/
  filter_upwards [eventually_gt_atTop 0] with n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : LT.lt 0 n
    ⊢ LT.lt 0 (AkraBazziRecurrence.asympBound g a b n)
  -/
  exact R.asympBound_pos n hn
  /-
    🎉 no goals
  -/


lemma eventually_asympBound_r_pos : ∀ᶠ (n : ℕ) in atTop, ∀ i, 0 < asympBound g a b (r i n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LT.lt 0 (AkraBazziRecurrence.asympBou …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LT.lt 0 (AkraBazziRecurrence.asympBou …
  -/
  exact fun i => (R.tendsto_atTop_r i).eventually R.eventually_asympBound_pos
  /-
    🎉 no goals
  -/


lemma eventually_atTop_sumTransform_le :
    ∃ c > 0, ∀ᶠ (n : ℕ) in atTop, ∀ i, sumTransform (p a b) g (r i n) n ≤ c * g n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  obtain ⟨c₁, hc₁_mem, hc₁⟩ := R.exists_eventually_const_mul_le_r
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  obtain ⟨c₂, hc₂_mem, hc₂⟩ := R.g_grows_poly.eventually_atTop_le_nat hc₁_mem
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  have hc₁_pos : 0 < c₁ := hc₁_mem.1
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    hc₁_pos : LT.lt 0 c₁
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  refine ⟨max c₂ (c₂ / c₁ ^ (p a b + 1)), by positivity, ?_⟩
  filter_upwards [hc₁, hc₂, R.eventually_r_pos, R.eventually_r_lt_n, eventually_gt_atTop 0]
    with n hn₁ hn₂ hrpos hr_lt_n hn_pos
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    hc₁_pos : LT.lt 0 c₁
    n : Nat
    hn₁ : ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)
    hn₂ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul c₁ ↑n) ↑n) u → LE.le (g …
    hrpos : ∀ (i : α), LT.lt 0 (r i n)
    hr_lt_n : ∀ (i : α), LT.lt (r i n) n
    hn_pos : LT.lt 0 n
    ⊢ ∀ (i : α), LE.le (AkraBazziRecurrence.sumTransform (AkraBazziRecurrence.p a  …
  -/
  intro i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    hc₁_pos : LT.lt 0 c₁
    n : Nat
    hn₁ : ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)
    hn₂ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul c₁ ↑n) ↑n) u → LE.le (g …
    hrpos : ∀ (i : α), LT.lt 0 (r i n)
    hr_lt_n : ∀ (i : α), LT.lt (r i n) n
    hn_pos : LT.lt 0 n
    i : α
    ⊢ LE.le (AkraBazziRecurrence.sumTransform (AkraBazziRecurrence.p a b) g (r i n …
  -/
  have hrpos_i := hrpos i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    hc₁_pos : LT.lt 0 c₁
    n : Nat
    hn₁ : ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)
    hn₂ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul c₁ ↑n) ↑n) u → LE.le (g …
    hrpos : ∀ (i : α), LT.lt 0 (r i n)
    hr_lt_n : ∀ (i : α), LT.lt (r i n) n
    hn_pos : LT.lt 0 n
    i : α
    hrpos_i : LT.lt 0 (r i n)
    ⊢ LE.le (AkraBazziRecurrence.sumTransform (AkraBazziRecurrence.p a b) g (r i n …
  -/
  have g_nonneg : 0 ≤ g n := R.g_nonneg n (by positivity)
  cases le_or_lt 0 (p a b + 1) with
  | inl hp => -- 0 ≤ p a b + 1
    calc sumTransform (p a b) g (r i n) n
           = n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1)) := by rfl
         _ ≤ n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, c₂ * g n / u ^ ((p a b) + 1)) := by
                gcongr with u hu
                rw [Finset.mem_Ico] at hu
                have hu' : u ∈ Set.Icc (r i n) n := ⟨hu.1, by omega⟩
                refine hn₂ u ?_
                rw [Set.mem_Icc]
                refine ⟨?_, by norm_cast; omega⟩
                calc c₁ * n ≤ r i n := by exact hn₁ i
                          _ ≤ u := by exact_mod_cast hu'.1
         _ ≤ n ^ (p a b) * (∑ _u ∈ Finset.Ico (r i n) n, c₂ * g n / (r i n) ^ ((p a b) + 1)) := by
                  gcongr with u hu; rw [Finset.mem_Ico] at hu; exact hu.1
         _ ≤ n ^ p a b * #(Ico (r i n) n) • (c₂ * g n / r i n ^ (p a b + 1)) := by
                  gcongr; exact Finset.sum_le_card_nsmul _ _ _ (fun x _ => by rfl)
         _ = n ^ p a b * #(Ico (r i n) n) * (c₂ * g n / r i n ^ (p a b + 1)) := by
                  rw [nsmul_eq_mul, mul_assoc]
         _ = n ^ (p a b) * (n - r i n) * (c₂ * g n / (r i n) ^ ((p a b) + 1)) := by
                  congr; rw [Nat.card_Ico, Nat.cast_sub (le_of_lt <| hr_lt_n i)]
         _ ≤ n ^ (p a b) * n * (c₂ * g n / (r i n) ^ ((p a b) + 1)) := by
                  gcongr; simp only [tsub_le_iff_right, le_add_iff_nonneg_right, Nat.cast_nonneg]
         _ ≤ n ^ (p a b) * n * (c₂ * g n / (c₁ * n) ^ ((p a b) + 1)) := by
                gcongr; exact hn₁ i
         _ = c₂ * g n * n ^ ((p a b) + 1) / (c₁ * n) ^ ((p a b) + 1) := by
                rw [← Real.rpow_add_one (by positivity) (p a b)]; ring
         _ = c₂ * g n * n ^ ((p a b) + 1) / (n ^ ((p a b) + 1) * c₁ ^ ((p a b) + 1)) := by
                rw [mul_comm c₁, Real.mul_rpow (by positivity) (by positivity)]
         _ = c₂ * g n * (n ^ ((p a b) + 1) / (n ^ ((p a b) + 1))) / c₁ ^ ((p a b) + 1) := by ring
         _ = c₂ * g n / c₁ ^ ((p a b) + 1) := by rw [div_self (by positivity), mul_one]
         _ = (c₂ / c₁ ^ ((p a b) + 1)) * g n := by ring
         _ ≤ max c₂ (c₂ / c₁ ^ ((p a b) + 1)) * g n := by gcongr; exact le_max_right _ _
  | inr hp => -- p a b + 1 < 0
    calc sumTransform (p a b) g (r i n) n
           = n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1)) := by rfl
         _ ≤ n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, c₂ * g n / u ^ ((p a b) + 1)) := by
                gcongr with u hu
                rw [Finset.mem_Ico] at hu
                have hu' : u ∈ Set.Icc (r i n) n := ⟨hu.1, by omega⟩
                refine hn₂ u ?_
                rw [Set.mem_Icc]
                refine ⟨?_, by norm_cast; omega⟩
                calc c₁ * n ≤ r i n := by exact hn₁ i
                          _ ≤ u     := by exact_mod_cast hu'.1
         _ ≤ n ^ (p a b) * (∑ _u ∈ Finset.Ico (r i n) n, c₂ * g n / n ^ ((p a b) + 1)) := by
                gcongr n ^ (p a b) * (Finset.Ico (r i n) n).sum (fun _ => c₂ * g n / ?_) with u hu
                rw [Finset.mem_Ico] at hu
                have : 0 < u := calc
                  0 < r i n := by exact hrpos_i
                  _ ≤ u := by exact hu.1
                exact rpow_le_rpow_of_exponent_nonpos (by positivity)
                  (by exact_mod_cast (le_of_lt hu.2)) (le_of_lt hp)
         _ ≤ n ^ p a b * #(Ico (r i n) n) • (c₂ * g n / n ^ (p a b + 1)) := by
                  gcongr; exact Finset.sum_le_card_nsmul _ _ _ (fun x _ => by rfl)
         _ = n ^ p a b * #(Ico (r i n) n) * (c₂ * g n / n ^ (p a b + 1)) := by
                  rw [nsmul_eq_mul, mul_assoc]
         _ = n ^ (p a b) * (n - r i n) * (c₂ * g n / n ^ ((p a b) + 1)) := by
                  congr; rw [Nat.card_Ico, Nat.cast_sub (le_of_lt <| hr_lt_n i)]
         _ ≤ n ^ (p a b) * n * (c₂ * g n / n ^ ((p a b) + 1)) := by
                gcongr; simp only [tsub_le_iff_right, le_add_iff_nonneg_right, Nat.cast_nonneg]
         _ = c₂ * (n^((p a b) + 1) / n ^ ((p a b) + 1)) * g n := by
                rw [← Real.rpow_add_one (by positivity) (p a b)]; ring
         _ = c₂ * g n := by rw [div_self (by positivity), mul_one]
         _ ≤ max c₂ (c₂ / c₁ ^ ((p a b) + 1)) * g n := by gcongr; exact le_max_left _ _


lemma eventually_atTop_sumTransform_ge :
    ∃ c > 0, ∀ᶠ (n : ℕ) in atTop, ∀ i, c * g n ≤ sumTransform (p a b) g (r i n) n := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  obtain ⟨c₁, hc₁_mem, hc₁⟩ := R.exists_eventually_const_mul_le_r
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  obtain ⟨c₂, hc₂_mem, hc₂⟩ := R.g_grows_poly.eventually_atTop_ge_nat hc₁_mem
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  obtain ⟨c₃, hc₃_mem, hc₃⟩ := R.exists_eventually_r_le_const_mul
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : Membership.mem (Set.Ioo 0 1) c₃
    hc₃ : Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  have hc₁_pos : 0 < c₁ := hc₁_mem.1
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : Membership.mem (Set.Ioo 0 1) c₃
    hc₃ : Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n …
    hc₁_pos : LT.lt 0 c₁
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  have hc₃' : 0 < (1 - c₃) := by have := hc₃_mem.2; linarith
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : Membership.mem (Set.Ioo 0 1) c₃
    hc₃ : Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n …
    hc₁_pos : LT.lt 0 c₁
    hc₃' : LT.lt 0 (HSub.hSub 1 c₃)
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (i : α), LE.l …
  -/
  refine ⟨min (c₂ * (1 - c₃)) ((1 - c₃) * c₂ / c₁^((p a b) + 1)), by positivity, ?_⟩
  filter_upwards [hc₁, hc₂, hc₃, R.eventually_r_pos, R.eventually_r_lt_n, eventually_gt_atTop 0]
    with n hn₁ hn₂ hn₃ hrpos hr_lt_n hn_pos
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : Membership.mem (Set.Ioo 0 1) c₃
    hc₃ : Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n …
    hc₁_pos : LT.lt 0 c₁
    hc₃' : LT.lt 0 (HSub.hSub 1 c₃)
    n : Nat
    hn₁ : ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)
    hn₂ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul c₁ ↑n) ↑n) u → LE.le (H …
    hn₃ : ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n)
    hrpos : ∀ (i : α), LT.lt 0 (r i n)
    hr_lt_n : ∀ (i : α), LT.lt (r i n) n
    hn_pos : LT.lt 0 n
    ⊢ ∀ (i : α), LE.le (HMul.hMul (Min.min (HMul.hMul c₂ (HSub.hSub 1 c₃)) (HDiv.h …
  -/
  intro i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : Membership.mem (Set.Ioo 0 1) c₃
    hc₃ : Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n …
    hc₁_pos : LT.lt 0 c₁
    hc₃' : LT.lt 0 (HSub.hSub 1 c₃)
    n : Nat
    hn₁ : ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)
    hn₂ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul c₁ ↑n) ↑n) u → LE.le (H …
    hn₃ : ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n)
    hrpos : ∀ (i : α), LT.lt 0 (r i n)
    hr_lt_n : ∀ (i : α), LT.lt (r i n) n
    hn_pos : LT.lt 0 n
    i : α
    ⊢ LE.le (HMul.hMul (Min.min (HMul.hMul c₂ (HSub.hSub 1 c₃)) (HDiv.hDiv (HMul.h …
  -/
  have hrpos_i := hrpos i
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    c₁ : Real
    hc₁_mem : Membership.mem (Set.Ioo 0 1) c₁
    hc₁ : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)) …
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hc₂ : Filter.Eventually (fun n => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : Membership.mem (Set.Ioo 0 1) c₃
    hc₃ : Filter.Eventually (fun n => ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n …
    hc₁_pos : LT.lt 0 c₁
    hc₃' : LT.lt 0 (HSub.hSub 1 c₃)
    n : Nat
    hn₁ : ∀ (i : α), LE.le (HMul.hMul c₁ ↑n) ↑(r i n)
    hn₂ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul c₁ ↑n) ↑n) u → LE.le (H …
    hn₃ : ∀ (i : α), LE.le (↑(r i n)) (HMul.hMul c₃ ↑n)
    hrpos : ∀ (i : α), LT.lt 0 (r i n)
    hr_lt_n : ∀ (i : α), LT.lt (r i n) n
    hn_pos : LT.lt 0 n
    i : α
    hrpos_i : LT.lt 0 (r i n)
    ⊢ LE.le (HMul.hMul (Min.min (HMul.hMul c₂ (HSub.hSub 1 c₃)) (HDiv.hDiv (HMul.h …
  -/
  have g_nonneg : 0 ≤ g n := R.g_nonneg n (by positivity)
  cases le_or_gt 0 (p a b + 1) with
  | inl hp => -- 0 ≤ (p a b) + 1
    calc sumTransform (p a b) g (r i n) n
           = n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1))    := rfl
         _ ≥ n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, c₂ * g n / u^((p a b) + 1)) := by
                gcongr with u hu
                rw [Finset.mem_Ico] at hu
                have hu' : u ∈ Set.Icc (r i n) n := ⟨hu.1, by omega⟩
                refine hn₂ u ?_
                rw [Set.mem_Icc]
                refine ⟨?_, by norm_cast; omega⟩
                calc c₁ * n ≤ r i n := by exact hn₁ i
                          _ ≤ u     := by exact_mod_cast hu'.1
         _ ≥ n ^ (p a b) * (∑ _u ∈ Finset.Ico (r i n) n, c₂ * g n / n ^ ((p a b) + 1)) := by
                gcongr with u hu
                · rw [Finset.mem_Ico] at hu
                  have := calc 0 < r i n := hrpos_i
                              _ ≤ u := hu.1
                  positivity
                · rw [Finset.mem_Ico] at hu
                  exact le_of_lt hu.2
         _ ≥ n ^ p a b * #(Ico (r i n) n) • (c₂ * g n / n ^ (p a b + 1)) := by
                gcongr; exact Finset.card_nsmul_le_sum _ _ _ (fun x _ => by rfl)
         _ = n ^ p a b * #(Ico (r i n) n) * (c₂ * g n / n ^ (p a b + 1)) := by
                rw [nsmul_eq_mul, mul_assoc]
         _ = n ^ (p a b) * (n - r i n) * (c₂ * g n / n ^ ((p a b) + 1)) := by
                congr; rw [Nat.card_Ico, Nat.cast_sub (le_of_lt <| hr_lt_n i)]
         _ ≥ n ^ (p a b) * (n - c₃ * n) * (c₂ * g n / n ^ ((p a b) + 1)) := by
                gcongr; exact hn₃ i
         _ = n ^ (p a b) * n * (1 - c₃) * (c₂ * g n / n ^ ((p a b) + 1)) := by ring
         _ = c₂ * (1 - c₃) * g n * (n ^ ((p a b) + 1) / n ^ ((p a b) + 1)) := by
                rw [← Real.rpow_add_one (by positivity) (p a b)]; ring
         _ = c₂ * (1 - c₃) * g n := by rw [div_self (by positivity), mul_one]
         _ ≥ min (c₂ * (1 - c₃)) ((1 - c₃) * c₂ / c₁ ^ ((p a b) + 1)) * g n := by
                gcongr; exact min_le_left _ _
  | inr hp => -- (p a b) + 1 < 0
    calc sumTransform (p a b) g (r i n) n
        = n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, g u / u^((p a b) + 1))        := by rfl
      _ ≥ n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, c₂ * g n / u ^ ((p a b) + 1)) := by
             gcongr with u hu
             rw [Finset.mem_Ico] at hu
             have hu' : u ∈ Set.Icc (r i n) n := ⟨hu.1, by omega⟩
             refine hn₂ u ?_
             rw [Set.mem_Icc]
             refine ⟨?_, by norm_cast; omega⟩
             calc c₁ * n ≤ r i n := by exact hn₁ i
                       _ ≤ u := by exact_mod_cast hu'.1
      _ ≥ n ^ (p a b) * (∑ _u ∈ Finset.Ico (r i n) n, c₂ * g n / (r i n) ^ ((p a b) + 1)) := by
             gcongr n^(p a b) * (Finset.Ico (r i n) n).sum (fun _ => c₂ * g n / ?_) with u hu
             · rw [Finset.mem_Ico] at hu
               have := calc 0 < r i n := hrpos_i
                           _ ≤ u := hu.1
               positivity
             · rw [Finset.mem_Ico] at hu
               exact rpow_le_rpow_of_exponent_nonpos (by positivity)
                 (by exact_mod_cast hu.1) (le_of_lt hp)
      _ ≥ n ^ p a b * #(Ico (r i n) n) • (c₂ * g n / r i n ^ (p a b + 1)) := by
             gcongr; exact Finset.card_nsmul_le_sum _ _ _ (fun x _ => by rfl)
      _ = n ^ p a b * #(Ico (r i n) n) * (c₂ * g n / r i n ^ (p a b + 1)) := by
             rw [nsmul_eq_mul, mul_assoc]
      _ ≥ n ^ p a b * #(Ico (r i n) n) * (c₂ * g n / (c₁ * n) ^ (p a b + 1)) := by
             gcongr n ^ p a b * #(Ico (r i n) n) * (c₂ * g n / ?_)
             exact rpow_le_rpow_of_exponent_nonpos (by positivity) (hn₁ i) (le_of_lt hp)
      _ = n ^ (p a b) * (n - r i n) * (c₂ * g n / (c₁ * n) ^ ((p a b) + 1)) := by
             congr; rw [Nat.card_Ico, Nat.cast_sub (le_of_lt <| hr_lt_n i)]
      _ ≥ n ^ (p a b) * (n - c₃ * n) * (c₂ * g n / (c₁ * n) ^ ((p a b) + 1)) := by
             gcongr; exact hn₃ i
      _ = n ^ (p a b) * n * (1 - c₃) * (c₂ * g n / (c₁ * n) ^ ((p a b) + 1)) := by ring
      _ = n ^ (p a b) * n * (1 - c₃) * (c₂ * g n / (c₁ ^ ((p a b) + 1) * n ^ ((p a b) + 1))) := by
             rw [Real.mul_rpow (by positivity) (by positivity)]
      _ = (n ^ ((p a b) + 1) / n ^ ((p a b) + 1)) * (1 - c₃) * c₂ * g n / c₁ ^ ((p a b) + 1) := by
             rw [← Real.rpow_add_one (by positivity) (p a b)]; ring
      _ = (1 - c₃) * c₂ / c₁ ^ ((p a b) + 1) * g n := by
             rw [div_self (by positivity), one_mul]; ring
      _ ≥ min (c₂ * (1 - c₃)) ((1 - c₃) * c₂ / c₁ ^ ((p a b) + 1)) * g n := by
             gcongr; exact min_le_right _ _


lemma eventually_deriv_rpow_p_mul_one_sub_smoothingFn (p : ℝ) :
    deriv (fun z => z ^ p * (1 - ε z))
      =ᶠ[atTop] fun z => p * z ^ (p-1) * (1 - ε z) + z ^ (p-1) / (log z ^ 2) := calc
  deriv (fun x => x ^ p * (1 - ε x))
    =ᶠ[atTop] fun x => deriv (· ^ p) x * (1 - ε x) + x ^ p * deriv (1 - ε ·) x := by
            /-
              p : Real
              ⊢ Filter.atTop.EventuallyEq (deriv fun x => HMul.hMul (HPow.hPow x p) (HSub.hS …
            -/
            filter_upwards [eventually_gt_atTop 1] with x hx
            /-
              case h
              p x : Real
              hx : LT.lt 1 x
              ⊢ Eq (deriv (fun x => HMul.hMul (HPow.hPow x p) (HSub.hSub 1 (AkraBazziRecurre …
            -/
            rw [deriv_mul]
              /-
                case h.hc
                p x : Real
                hx : LT.lt 1 x
                ⊢ DifferentiableAt Real (fun x => HPow.hPow x p) x
              -/
            · exact differentiableAt_rpow_const_of_ne _ (by positivity)
              /-
                🎉 no goals
              -/
              /-
                case h.hd
                p x : Real
                hx : LT.lt 1 x
                ⊢ DifferentiableAt Real (fun x => HSub.hSub 1 (AkraBazziRecurrence.smoothingFn …
              -/
            · exact differentiableAt_one_sub_smoothingFn hx
              /-
                🎉 no goals
              -/
  _ =ᶠ[atTop] fun x => p * x ^ (p-1) * (1 - ε x) + x ^ p * (x⁻¹ / (log x ^ 2)) := by
            filter_upwards [eventually_gt_atTop 1, eventually_deriv_one_sub_smoothingFn]
              with x hx hderiv
            /-
              case h
              p x : Real
              hx : LT.lt 1 x
              hderiv : Eq (deriv (fun x => HSub.hSub 1 (AkraBazziRecurrence.smoothingFn x))  …
              ⊢ Eq (HAdd.hAdd (HMul.hMul (deriv (fun x => HPow.hPow x p) x) (HSub.hSub 1 (Ak …
            -/
            rw [hderiv, Real.deriv_rpow_const (Or.inl <| by positivity)]
            /-
              🎉 no goals
            -/
  _ =ᶠ[atTop] fun x => p * x ^ (p-1) * (1 - ε x) + x ^ (p-1) / (log x ^ 2) := by
            /-
              p : Real
              ⊢ Filter.atTop.EventuallyEq (fun x => HAdd.hAdd (HMul.hMul (HMul.hMul p (HPow. …
            -/
            filter_upwards [eventually_gt_atTop 0] with x hx
            /-
              case h
              p x : Real
              hx : LT.lt 0 x
              ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul p (HPow.hPow x (HSub.hSub p 1))) (HSub.h …
            -/
            rw [mul_div, ← Real.rpow_neg_one, ← Real.rpow_add (by positivity), sub_eq_add_neg]
            /-
              🎉 no goals
            -/


lemma eventually_deriv_rpow_p_mul_one_add_smoothingFn (p : ℝ) :
    deriv (fun z => z ^ p * (1 + ε z))
      =ᶠ[atTop] fun z => p * z ^ (p-1) * (1 + ε z) - z ^ (p-1) / (log z ^ 2) := calc
  deriv (fun x => x ^ p * (1 + ε x))
    =ᶠ[atTop] fun x => deriv (· ^ p) x * (1 + ε x) + x ^ p * deriv (1 + ε ·) x := by
            /-
              p : Real
              ⊢ Filter.atTop.EventuallyEq (deriv fun x => HMul.hMul (HPow.hPow x p) (HAdd.hA …
            -/
            filter_upwards [eventually_gt_atTop 1] with x hx
            /-
              case h
              p x : Real
              hx : LT.lt 1 x
              ⊢ Eq (deriv (fun x => HMul.hMul (HPow.hPow x p) (HAdd.hAdd 1 (AkraBazziRecurre …
            -/
            rw [deriv_mul]
              /-
                case h.hc
                p x : Real
                hx : LT.lt 1 x
                ⊢ DifferentiableAt Real (fun x => HPow.hPow x p) x
              -/
            · exact differentiableAt_rpow_const_of_ne _ (by positivity)
              /-
                🎉 no goals
              -/
              /-
                case h.hd
                p x : Real
                hx : LT.lt 1 x
                ⊢ DifferentiableAt Real (fun x => HAdd.hAdd 1 (AkraBazziRecurrence.smoothingFn …
              -/
            · exact differentiableAt_one_add_smoothingFn hx
              /-
                🎉 no goals
              -/
  _ =ᶠ[atTop] fun x => p * x ^ (p-1) * (1 + ε x) - x ^ p * (x⁻¹ / (log x ^ 2)) := by
            filter_upwards [eventually_gt_atTop 1, eventually_deriv_one_add_smoothingFn]
              with x hx hderiv
            /-
              case h
              p x : Real
              hx : LT.lt 1 x
              hderiv : Eq (deriv (fun x => HAdd.hAdd 1 (AkraBazziRecurrence.smoothingFn x))  …
              ⊢ Eq (HAdd.hAdd (HMul.hMul (deriv (fun x => HPow.hPow x p) x) (HAdd.hAdd 1 (Ak …
            -/
            simp [hderiv, Real.deriv_rpow_const (Or.inl <| by positivity), neg_div, sub_eq_add_neg]
            /-
              🎉 no goals
            -/
  _ =ᶠ[atTop] fun x => p * x ^ (p-1) * (1 + ε x) - x ^ (p-1) / (log x ^ 2) := by
            /-
              p : Real
              ⊢ Filter.atTop.EventuallyEq (fun x => HSub.hSub (HMul.hMul (HMul.hMul p (HPow. …
            -/
            filter_upwards [eventually_gt_atTop 0] with x hx
            /-
              case h
              p x : Real
              hx : LT.lt 0 x
              ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul p (HPow.hPow x (HSub.hSub p 1))) (HAdd.h …
            -/
            simp [mul_div, ← Real.rpow_neg_one, ← Real.rpow_add (by positivity), sub_eq_add_neg]
            /-
              🎉 no goals
            -/


lemma isEquivalent_deriv_rpow_p_mul_one_sub_smoothingFn {p : ℝ} (hp : p ≠ 0) :
    deriv (fun z => z ^ p * (1 - ε z)) ~[atTop] fun z => p * z ^ (p-1) := calc
  deriv (fun z => z ^ p * (1 - ε z))
    =ᶠ[atTop] fun z => p * z ^ (p-1) * (1 - ε z) + z^(p-1) / (log z ^ 2) :=
        eventually_deriv_rpow_p_mul_one_sub_smoothingFn p
  _ ~[atTop] fun z => p * z ^ (p-1) := by
        /-
          p : Real
          hp : Ne p 0
          ⊢ Asymptotics.IsEquivalent Filter.atTop (fun z => HAdd.hAdd (HMul.hMul (HMul.h …
        -/
        refine IsEquivalent.add_isLittleO ?one ?two
        case one => calc
          (fun z => p * z ^ (p-1) * (1 - ε z)) ~[atTop] fun z => p * z ^ (p-1) * 1 :=
                IsEquivalent.mul IsEquivalent.refl isEquivalent_one_sub_smoothingFn_one
          _ = fun z => p * z ^ (p-1) := by ext; ring
        case two => calc
          (fun z => z ^ (p-1) / (log z ^ 2)) =o[atTop] fun z => z ^ (p-1) / 1 := by
                      simp_rw [div_eq_mul_inv]
                      refine IsBigO.mul_isLittleO (isBigO_refl _ _)
                        (IsLittleO.inv_rev ?_ (by aesop (add safe Eventually.of_forall)))
                      rw [isLittleO_const_left]
                      refine Or.inr <| Tendsto.comp tendsto_norm_atTop_atTop ?_
                      exact Tendsto.comp (g := fun z => z ^ 2)
                        (tendsto_pow_atTop (by norm_num)) tendsto_log_atTop
          _ = fun z => z ^ (p-1) := by ext; simp
          _ =Θ[atTop] fun z => p * z ^ (p-1) := by
                      exact IsTheta.const_mul_right hp <| isTheta_refl _ _


lemma isEquivalent_deriv_rpow_p_mul_one_add_smoothingFn {p : ℝ} (hp : p ≠ 0) :
    deriv (fun z => z ^ p * (1 + ε z)) ~[atTop] fun z => p * z ^ (p-1) := calc
  deriv (fun z => z ^ p * (1 + ε z))
    =ᶠ[atTop] fun z => p * z ^ (p-1) * (1 + ε z) - z ^ (p-1) / (log z ^ 2) :=
        eventually_deriv_rpow_p_mul_one_add_smoothingFn p
  _ ~[atTop] fun z => p * z ^ (p-1) := by
        /-
          p : Real
          hp : Ne p 0
          ⊢ Asymptotics.IsEquivalent Filter.atTop (fun z => HSub.hSub (HMul.hMul (HMul.h …
        -/
        refine IsEquivalent.add_isLittleO ?one ?two
        case one => calc
          (fun z => p * z ^ (p-1) * (1 + ε z)) ~[atTop] fun z => p * z ^ (p-1) * 1 :=
                IsEquivalent.mul IsEquivalent.refl isEquivalent_one_add_smoothingFn_one
          _ = fun z => p * z ^ (p-1) := by ext; ring
        case two => calc
          (fun z => -(z ^ (p-1) / (log z ^ 2))) =o[atTop] fun z => z ^ (p-1) / 1 := by
                      simp_rw [isLittleO_neg_left, div_eq_mul_inv]
                      refine IsBigO.mul_isLittleO (isBigO_refl _ _)
                        (IsLittleO.inv_rev ?_ (by aesop (add safe Eventually.of_forall)))
                      rw [isLittleO_const_left]
                      refine Or.inr <| Tendsto.comp tendsto_norm_atTop_atTop ?_
                      exact Tendsto.comp (g := fun z => z ^ 2)
                        (tendsto_pow_atTop (by norm_num)) tendsto_log_atTop
          _ = fun z => z ^ (p-1) := by ext; simp
          _ =Θ[atTop] fun z => p * z ^ (p-1) := by
                      exact IsTheta.const_mul_right hp <| isTheta_refl _ _


lemma isTheta_deriv_rpow_p_mul_one_sub_smoothingFn {p : ℝ} (hp : p ≠ 0) :
    (fun x => ‖deriv (fun z => z ^ p * (1 - ε z)) x‖) =Θ[atTop] fun z => z ^ (p-1) := by
  /-
    p : Real
    hp : Ne p 0
    ⊢ Asymptotics.IsTheta Filter.atTop (fun x => Norm.norm (deriv (fun z => HMul.h …
  -/
  refine IsTheta.norm_left ?_
  calc (fun x => deriv (fun z => z ^ p * (1 - ε z)) x) =Θ[atTop] fun z => p * z ^ (p-1) :=
            (isEquivalent_deriv_rpow_p_mul_one_sub_smoothingFn hp).isTheta
    _ =Θ[atTop] fun z => z ^ (p-1) :=
            IsTheta.const_mul_left hp <| isTheta_refl _ _


lemma isTheta_deriv_rpow_p_mul_one_add_smoothingFn {p : ℝ} (hp : p ≠ 0) :
    (fun x => ‖deriv (fun z => z ^ p * (1 + ε z)) x‖) =Θ[atTop] fun z => z ^ (p-1) := by
  /-
    p : Real
    hp : Ne p 0
    ⊢ Asymptotics.IsTheta Filter.atTop (fun x => Norm.norm (deriv (fun z => HMul.h …
  -/
  refine IsTheta.norm_left ?_
  calc (fun x => deriv (fun z => z ^ p * (1 + ε z)) x) =Θ[atTop] fun z => p * z ^ (p-1) :=
            (isEquivalent_deriv_rpow_p_mul_one_add_smoothingFn hp).isTheta
    _ =Θ[atTop] fun z => z ^ (p-1) :=
            IsTheta.const_mul_left hp <| isTheta_refl _ _


lemma growsPolynomially_deriv_rpow_p_mul_one_sub_smoothingFn (p : ℝ) :
    GrowsPolynomially fun x => ‖deriv (fun z => z ^ p * (1 - ε z)) x‖ := by
  cases eq_or_ne p 0 with
  | inl hp => -- p = 0
    have h₁ : (fun x => ‖deriv (fun z => z ^ p * (1 - ε z)) x‖)
        =ᶠ[atTop] fun z => z⁻¹ / (log z ^ 2) := by
      filter_upwards [eventually_deriv_one_sub_smoothingFn, eventually_gt_atTop 1] with x hx hx_pos
      have : 0 ≤ x⁻¹ / (log x ^ 2) := by
        have hlog : 0 < log x := Real.log_pos hx_pos
        positivity
      simp only [hp, Real.rpow_zero, one_mul, differentiableAt_const, hx, Real.norm_of_nonneg this]
    refine GrowsPolynomially.congr_of_eventuallyEq h₁ ?_
    refine GrowsPolynomially.div (GrowsPolynomially.inv growsPolynomially_id)
      (GrowsPolynomially.pow 2 growsPolynomially_log ?_)
    filter_upwards [eventually_ge_atTop 1] with _ hx
    exact log_nonneg hx
  | inr hp => -- p ≠ 0
    refine GrowsPolynomially.of_isTheta (growsPolynomially_rpow (p-1))
      (isTheta_deriv_rpow_p_mul_one_sub_smoothingFn hp) ?_
    filter_upwards [eventually_gt_atTop 0] with _ _
    positivity


lemma growsPolynomially_deriv_rpow_p_mul_one_add_smoothingFn (p : ℝ) :
    GrowsPolynomially fun x => ‖deriv (fun z => z ^ p * (1 + ε z)) x‖ := by
  cases eq_or_ne p 0 with
  | inl hp => -- p = 0
    have h₁ : (fun x => ‖deriv (fun z => z ^ p * (1 + ε z)) x‖)
        =ᶠ[atTop] fun z => z⁻¹ / (log z ^ 2) := by
      filter_upwards [eventually_deriv_one_add_smoothingFn, eventually_gt_atTop 1] with x hx hx_pos
      have : 0 ≤ x⁻¹ / (log x ^ 2) := by
        have hlog : 0 < log x := Real.log_pos hx_pos
        positivity
      simp only [neg_div, norm_neg, hp, Real.rpow_zero,
        one_mul, differentiableAt_const, hx, Real.norm_of_nonneg this]
    refine GrowsPolynomially.congr_of_eventuallyEq h₁ ?_
    refine GrowsPolynomially.div (GrowsPolynomially.inv growsPolynomially_id)
      (GrowsPolynomially.pow 2 growsPolynomially_log ?_)
    filter_upwards [eventually_ge_atTop 1] with x hx
    exact log_nonneg hx
  | inr hp => -- p ≠ 0
    refine GrowsPolynomially.of_isTheta (growsPolynomially_rpow (p-1))
      (isTheta_deriv_rpow_p_mul_one_add_smoothingFn hp) ?_
    filter_upwards [eventually_gt_atTop 0] with _ _
    positivity


lemma isBigO_apply_r_sub_b (q : ℝ → ℝ) (hq_diff : DifferentiableOn ℝ q (Set.Ioi 1))
    (hq_poly : GrowsPolynomially fun x => ‖deriv q x‖) (i : α) :
    (fun n => q (r i n) - q (b i * n)) =O[atTop] fun n => (deriv q n) * (r i n - b i * n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.hM …
  -/
  let b' := b (min_bi b) / 2
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.hM …
  -/
  have hb_pos : 0 < b' := by have := R.b_pos (min_bi b); positivity
  have hb_lt_one : b' < 1 := calc
    b (min_bi b) / 2 < b (min_bi b) := by exact div_two_lt_of_pos (R.b_pos (min_bi b))
                   _ < 1 := R.b_lt_one (min_bi b)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    hb_lt_one : LT.lt b' 1
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.hM …
  -/
  have hb : b' ∈ Set.Ioo 0 1 := ⟨hb_pos, hb_lt_one⟩
  have hb' : ∀ i, b' ≤ b i := fun i => calc
    b (min_bi b) / 2 ≤ b i / 2 := by gcongr; aesop
               _ ≤ b i := by exact le_of_lt <| div_two_lt_of_pos (R.b_pos i)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    hb_lt_one : LT.lt b' 1
    hb : Membership.mem (Set.Ioo 0 1) b'
    hb' : ∀ (i : α), LE.le b' (b i)
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.hM …
  -/
  obtain ⟨c₁, _, c₂, _, hq_poly⟩ := hq_poly b' hb
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly✝ : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    hb_lt_one : LT.lt b' 1
    hb : Membership.mem (Set.Ioo 0 1) b'
    hb' : ∀ (i : α), LE.le b' (b i)
    c₁ : Real
    left✝¹ : GT.gt c₁ 0
    c₂ : Real
    left✝ : GT.gt c₂ 0
    hq_poly : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (H …
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.hM …
  -/
  rw [isBigO_iff]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly✝ : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    hb_lt_one : LT.lt b' 1
    hb : Membership.mem (Set.Ioo 0 1) b'
    hb' : ∀ (i : α), LE.le b' (b i)
    c₁ : Real
    left✝¹ : GT.gt c₁ 0
    c₂ : Real
    left✝ : GT.gt c₂ 0
    hq_poly : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (H …
    ⊢ Exists fun c => Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (q ↑ …
  -/
  refine ⟨c₂, ?_⟩
  have h_tendsto : Tendsto (fun x => b' * x) atTop atTop :=
    Tendsto.const_mul_atTop hb_pos tendsto_id
  filter_upwards [hq_poly.natCast_atTop, R.eventually_bi_mul_le_r, eventually_ge_atTop R.n₀,
                  eventually_gt_atTop 0, (h_tendsto.eventually_gt_atTop 1).natCast_atTop] with
    n hn h_bi_le_r h_ge_n₀ h_n_pos h_bn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    q : Real → Real
    hq_diff : DifferentiableOn Real q (Set.Ioi 1)
    hq_poly✝ : AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (deriv q x)
    i : α
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    hb_lt_one : LT.lt b' 1
    hb : Membership.mem (Set.Ioo 0 1) b'
    hb' : ∀ (i : α), LE.le b' (b i)
    c₁ : Real
    left✝¹ : GT.gt c₁ 0
    c₂ : Real
    left✝ : GT.gt c₂ 0
    hq_poly : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (H …
    h_tendsto : Filter.Tendsto (fun x => HMul.hMul b' x) Filter.atTop Filter.atTop
    n : Nat
    hn : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b' ↑n) ↑n) u → Membershi …
    h_bi_le_r : ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv (b (AkraBazziRecurrence.min …
    h_ge_n₀ : LE.le R.n₀ n
    h_n_pos : LT.lt 0 n
    h_bn : LT.lt 1 (HMul.hMul b' ↑n)
    ⊢ LE.le (Norm.norm (HSub.hSub (q ↑(r i n)) (q (HMul.hMul (b i) ↑n)))) (HMul.hM …
  -/
  rw [norm_mul, ← mul_assoc]
  refine Convex.norm_image_sub_le_of_norm_deriv_le
    (s := Set.Icc (b'*n) n) (fun z hz => ?diff) (fun z hz => (hn z hz).2)
    (convex_Icc _ _) ?mem_Icc <| ⟨h_bi_le_r i, by exact_mod_cast (le_of_lt (R.r_lt_n i n h_ge_n₀))⟩
  case diff =>
    refine hq_diff.differentiableAt (Ioi_mem_nhds ?_)
    calc 1 < b' * n := by exact h_bn
         _ ≤ z := hz.1
  case mem_Icc =>
    refine ⟨by gcongr; exact hb' i, ?_⟩
    calc b i * n ≤ 1 * n := by gcongr; exact le_of_lt <| R.b_lt_one i
                 _ = n := by simp


lemma rpow_p_mul_one_sub_smoothingFn_le :
    ∀ᶠ (n : ℕ) in atTop, ∀ i, (r i n) ^ (p a b) * (1 - ε (r i n))
      ≤ (b i) ^ (p a b) * n ^ (p a b) * (1 - ε n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r i n) …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LE.le (HMul.hMul (HPow.hPow (↑(r i x) …
  -/
  intro i
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (HPow.hPow (↑(r i x)) (AkraBazz …
  -/
  let q : ℝ → ℝ := fun x => x ^ (p a b) * (1 - ε x)
  have h_diff_q : DifferentiableOn ℝ q (Set.Ioi 1) := by
    refine DifferentiableOn.mul
      (DifferentiableOn.mono (differentiableOn_rpow_const _) fun z hz => ?_)
        differentiableOn_one_sub_smoothingFn
    rw [Set.mem_compl_singleton_iff]
    rw [Set.mem_Ioi] at hz
    exact ne_of_gt <| zero_lt_one.trans hz
  have h_deriv_q : deriv q =O[atTop] fun x => x ^ ((p a b) - 1) := calc
    deriv q = deriv fun x => (fun z => z ^ (p a b)) x * (fun z => 1 - ε z) x := by rfl
          _ =ᶠ[atTop] fun x => deriv (fun z => z ^ (p a b)) x * (1 - ε x) +
                  x ^ (p a b) * deriv (fun z => 1 - ε z) x := by
              filter_upwards [eventually_ne_atTop 0, eventually_gt_atTop 1] with x hx hx'
              rw [deriv_mul] <;> aesop
          _ =O[atTop] fun x => x ^ ((p a b) - 1) := by
              refine IsBigO.add ?left ?right
              case left => calc
                (fun x => deriv (fun z => z ^ (p a b)) x * (1 - ε x))
                    =O[atTop] fun x => x ^ ((p a b) - 1) * (1 - ε x) := by
                      exact IsBigO.mul (isBigO_deriv_rpow_const_atTop (p a b)) (isBigO_refl _ _)
                  _ =O[atTop] fun x => x ^ ((p a b) - 1) * 1 := by
                      refine IsBigO.mul (isBigO_refl _ _)
                        isEquivalent_one_sub_smoothingFn_one.isBigO
                  _ = fun x => x ^ ((p a b) - 1) := by ext; rw [mul_one]
              case right => calc
                (fun x => x ^ (p a b) * deriv (fun z => 1 - ε z) x)
                    =O[atTop] (fun x => x ^ (p a b) * x⁻¹) := by
                      exact IsBigO.mul (isBigO_refl _ _) isLittleO_deriv_one_sub_smoothingFn.isBigO
                  _ =ᶠ[atTop] fun x => x ^ ((p a b) - 1) := by
                      filter_upwards [eventually_gt_atTop 0] with x hx
                      rw [← Real.rpow_neg_one, ← Real.rpow_add hx, ← sub_eq_add_neg]
  have h_main_norm : (fun (n : ℕ) => ‖q (r i n) - q (b i * n)‖)
      ≤ᶠ[atTop] fun (n : ℕ) => ‖(b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n)‖ := by
    refine IsLittleO.eventuallyLE ?_
    calc
      (fun (n : ℕ) => q (r i n) - q (b i * n))
          =O[atTop] fun n => (deriv q n) * (r i n - b i * n) := by
              exact R.isBigO_apply_r_sub_b q h_diff_q
                (growsPolynomially_deriv_rpow_p_mul_one_sub_smoothingFn (p a b)) i
        _ =o[atTop] fun n => (deriv q n) * (n / log n ^ 2) := by
              exact IsBigO.mul_isLittleO (isBigO_refl _ _) (R.dist_r_b i)
        _ =O[atTop] fun n => n^((p a b) - 1) * (n / log n ^ 2) := by
              exact IsBigO.mul (IsBigO.natCast_atTop h_deriv_q) (isBigO_refl _ _)
        _ =ᶠ[atTop] fun n => n^(p a b) / (log n) ^ 2 := by
              filter_upwards [eventually_ne_atTop 0] with n hn
              have hn' : (n : ℝ) ≠ 0 := by positivity
              simp [← mul_div_assoc, ← Real.rpow_add_one hn']
        _ = fun (n : ℕ) => (n : ℝ) ^ (p a b) * (1 / (log n)^2) := by
              simp_rw [mul_div, mul_one]
        _ =Θ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (1 / (log n)^2) := by
              refine IsTheta.symm ?_
              simp_rw [mul_assoc]
              refine IsTheta.const_mul_left ?_ (isTheta_refl _ _)
              have := R.b_pos i; positivity
        _ =Θ[atTop] fun (n : ℕ) => (b i)^(p a b) * n^(p a b) * (ε (b i * n) - ε n) := by
              exact IsTheta.symm <| IsTheta.mul (isTheta_refl _ _)
                <| R.isTheta_smoothingFn_sub_self i
  have h_main : (fun (n : ℕ) => q (r i n) - q (b i * n))
      ≤ᶠ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n) := by
    calc (fun (n : ℕ) => q (r i n) - q (b i * n))
           ≤ᶠ[atTop] fun (n : ℕ) => ‖q (r i n) - q (b i * n)‖ := by
                filter_upwards with _; exact le_norm_self _
         _ ≤ᶠ[atTop] fun (n : ℕ) => ‖(b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n)‖ :=
                h_main_norm
         _ =ᶠ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n) := by
                filter_upwards [eventually_gt_atTop ⌈(b i)⁻¹⌉₊, eventually_gt_atTop 1] with n hn hn'
                refine norm_of_nonneg ?_
                have h₁ := R.b_pos i
                have h₂ : 0 ≤ ε (b i * n) - ε n := by
                  refine sub_nonneg_of_le <|
                    (strictAntiOn_smoothingFn.le_iff_le ?n_gt_one ?bn_gt_one).mpr ?le
                  case n_gt_one =>
                    rwa [Set.mem_Ioi, Nat.one_lt_cast]
                  case bn_gt_one =>
                    calc 1 = b i * (b i)⁻¹ := by rw [mul_inv_cancel₀ (by positivity)]
                        _ ≤ b i * ⌈(b i)⁻¹⌉₊ := by gcongr; exact Nat.le_ceil _
                        _ < b i * n := by gcongr
                  case le => calc b i * n ≤ 1 * n := by have := R.b_lt_one i; gcongr
                                          _ = n := by rw [one_mul]
                positivity
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.h …
    ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (HPow.hPow (↑(r i x)) (AkraBazz …
  -/
  filter_upwards [h_main] with n hn
  have h₁ : q (b i * n) + (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n)
      = (b i) ^ (p a b) * n ^ (p a b) * (1 - ε n) := by
    have := R.b_pos i
    simp only [q, mul_rpow (by positivity : (0 : ℝ) ≤ b i) (by positivity : (0 : ℝ) ≤ n)]
    ring
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.h …
    n : Nat
    hn : LE.le (HSub.hSub (q ↑(r i n)) (q (HMul.hMul (b i) ↑n))) (HMul.hMul (HMul. …
    h₁ : Eq (HAdd.hAdd (q (HMul.hMul (b i) ↑n)) (HMul.hMul (HMul.hMul (HPow.hPow ( …
    ⊢ LE.le (HMul.hMul (HPow.hPow (↑(r i n)) (AkraBazziRecurrence.p a b)) (HSub.hS …
  -/
  show q (r i n) ≤ (b i) ^ (p a b) * n ^ (p a b) * (1 - ε n)
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.h …
    n : Nat
    hn : LE.le (HSub.hSub (q ↑(r i n)) (q (HMul.hMul (b i) ↑n))) (HMul.hMul (HMul. …
    h₁ : Eq (HAdd.hAdd (q (HMul.hMul (b i) ↑n)) (HMul.hMul (HMul.hMul (HPow.hPow ( …
    ⊢ LE.le (q ↑(r i n)) (HMul.hMul (HMul.hMul (HPow.hPow (b i) (AkraBazziRecurren …
  -/
  rw [← h₁, ← sub_le_iff_le_add']
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q ↑(r i n)) (q (HMul.h …
    n : Nat
    hn : LE.le (HSub.hSub (q ↑(r i n)) (q (HMul.hMul (b i) ↑n))) (HMul.hMul (HMul. …
    h₁ : Eq (HAdd.hAdd (q (HMul.hMul (b i) ↑n)) (HMul.hMul (HMul.hMul (HPow.hPow ( …
    ⊢ LE.le (HSub.hSub (q ↑(r i n)) (q (HMul.hMul (b i) ↑n))) (HMul.hMul (HMul.hMu …
  -/
  exact hn
  /-
    🎉 no goals
  -/


lemma rpow_p_mul_one_add_smoothingFn_ge :
    ∀ᶠ (n : ℕ) in atTop, ∀ i, (b i) ^ (p a b) * n ^ (p a b) * (1 + ε n)
      ≤ (r i n) ^ (p a b) * (1 + ε (r i n)) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPow.hPo …
  -/
  rw [Filter.eventually_all]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ ∀ (i : α), Filter.Eventually (fun x => LE.le (HMul.hMul (HMul.hMul (HPow.hPo …
  -/
  intro i
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (HMul.hMul (HPow.hPow (b i) (Ak …
  -/
  let q : ℝ → ℝ := fun x => x ^ (p a b) * (1 + ε x)
  have h_diff_q : DifferentiableOn ℝ q (Set.Ioi 1) := by
    refine DifferentiableOn.mul
        (DifferentiableOn.mono (differentiableOn_rpow_const _) fun z hz => ?_)
        differentiableOn_one_add_smoothingFn
    rw [Set.mem_compl_singleton_iff]
    rw [Set.mem_Ioi] at hz
    exact ne_of_gt <| zero_lt_one.trans hz
  have h_deriv_q : deriv q =O[atTop] fun x => x ^ ((p a b) - 1) := calc
    deriv q = deriv fun x => (fun z => z ^ (p a b)) x * (fun z => 1 + ε z) x := by rfl
          _ =ᶠ[atTop] fun x => deriv (fun z => z ^ (p a b)) x * (1 + ε x)
              + x ^ (p a b) * deriv (fun z => 1 + ε z) x := by
                filter_upwards [eventually_ne_atTop 0, eventually_gt_atTop 1] with x hx hx'
                rw [deriv_mul] <;> aesop
          _ =O[atTop] fun x => x ^ ((p a b) - 1) := by
                refine IsBigO.add ?left ?right
                case left => calc
                  (fun x => deriv (fun z => z ^ (p a b)) x * (1 + ε x))
                      =O[atTop] fun x => x ^ ((p a b) - 1) * (1 + ε x) := by
                        exact IsBigO.mul (isBigO_deriv_rpow_const_atTop (p a b)) (isBigO_refl _ _)
                    _ =O[atTop] fun x => x ^ ((p a b) - 1) * 1 :=
                        IsBigO.mul (isBigO_refl _ _) isEquivalent_one_add_smoothingFn_one.isBigO
                    _ = fun x => x ^ ((p a b) - 1) := by ext; rw [mul_one]
                case right => calc
                  (fun x => x ^ (p a b) * deriv (fun z => 1 + ε z) x)
                      =O[atTop] (fun x => x ^ (p a b) * x⁻¹) := by
                        exact IsBigO.mul (isBigO_refl _ _)
                          isLittleO_deriv_one_add_smoothingFn.isBigO
                    _ =ᶠ[atTop] fun x => x ^ ((p a b) - 1) := by
                        filter_upwards [eventually_gt_atTop 0] with x hx
                        rw [← Real.rpow_neg_one, ← Real.rpow_add hx, ← sub_eq_add_neg]
  have h_main_norm : (fun (n : ℕ) => ‖q (r i n) - q (b i * n)‖)
      ≤ᶠ[atTop] fun (n : ℕ) => ‖(b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n)‖ := by
    refine IsLittleO.eventuallyLE ?_
    calc
      (fun (n : ℕ) => q (r i n) - q (b i * n))
          =O[atTop] fun n => (deriv q n) * (r i n - b i * n) := by
            exact R.isBigO_apply_r_sub_b q h_diff_q
              (growsPolynomially_deriv_rpow_p_mul_one_add_smoothingFn (p a b)) i
        _ =o[atTop] fun n => (deriv q n) * (n / log n ^ 2) := by
            exact IsBigO.mul_isLittleO (isBigO_refl _ _) (R.dist_r_b i)
        _ =O[atTop] fun n => n ^ ((p a b) - 1) * (n / log n ^ 2) := by
            exact IsBigO.mul (IsBigO.natCast_atTop h_deriv_q) (isBigO_refl _ _)
        _ =ᶠ[atTop] fun n => n ^ (p a b) / (log n) ^ 2 := by
            filter_upwards [eventually_ne_atTop 0] with n hn
            have hn' : (n : ℝ) ≠ 0 := by positivity
            simp [← mul_div_assoc, ← Real.rpow_add_one hn']
        _ = fun (n : ℕ) => (n : ℝ) ^ (p a b) * (1 / (log n) ^ 2) := by simp_rw [mul_div, mul_one]
        _ =Θ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (1 / (log n) ^ 2) := by
            refine IsTheta.symm ?_
            simp_rw [mul_assoc]
            refine IsTheta.const_mul_left ?_ (isTheta_refl _ _)
            have := R.b_pos i; positivity
        _ =Θ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n) := by
            exact IsTheta.symm <| IsTheta.mul (isTheta_refl _ _)
                  <| R.isTheta_smoothingFn_sub_self i
  have h_main : (fun (n : ℕ) => q (b i * n) - q (r i n))
      ≤ᶠ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n) := by
    calc (fun (n : ℕ) => q (b i * n) - q (r i n))
           ≤ᶠ[atTop] fun (n : ℕ) => ‖q (r i n) - q (b i * n)‖ := by
              filter_upwards with _; rw [norm_sub_rev]; exact le_norm_self _
         _ ≤ᶠ[atTop] fun (n : ℕ) => ‖(b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n)‖ :=
              h_main_norm
         _ =ᶠ[atTop] fun (n : ℕ) => (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n) := by
              filter_upwards [eventually_gt_atTop ⌈(b i)⁻¹⌉₊, eventually_gt_atTop 1] with n hn hn'
              refine norm_of_nonneg ?_
              have h₁ := R.b_pos i
              have h₂ : 0 ≤ ε (b i * n) - ε n := by
                refine sub_nonneg_of_le <|
                  (strictAntiOn_smoothingFn.le_iff_le ?n_gt_one ?bn_gt_one).mpr ?le
                case n_gt_one =>
                  show 1 < (n : ℝ)
                  rw [Nat.one_lt_cast]
                  exact hn'
                case bn_gt_one =>
                  calc 1 = b i * (b i)⁻¹ := by rw [mul_inv_cancel₀ (by positivity)]
                      _ ≤ b i * ⌈(b i)⁻¹⌉₊ := by gcongr; exact Nat.le_ceil _
                      _ < b i * n := by gcongr
                case le => calc b i * n ≤ 1 * n := by have := R.b_lt_one i; gcongr
                                        _ = n := by rw [one_mul]
              positivity
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q (HMul.hMul (b i) ↑n) …
    ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (HMul.hMul (HPow.hPow (b i) (Ak …
  -/
  filter_upwards [h_main] with n hn
  have h₁ : q (b i * n) - (b i) ^ (p a b) * n ^ (p a b) * (ε (b i * n) - ε n)
      = (b i) ^ (p a b) * n ^ (p a b) * (1 + ε n) := by
    have := R.b_pos i
    simp only [q, mul_rpow (by positivity : (0 : ℝ) ≤ b i) (by positivity : (0 : ℝ) ≤ n)]
    ring
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q (HMul.hMul (b i) ↑n) …
    n : Nat
    hn : LE.le (HSub.hSub (q (HMul.hMul (b i) ↑n)) (q ↑(r i n))) (HMul.hMul (HMul. …
    h₁ : Eq (HSub.hSub (q (HMul.hMul (b i) ↑n)) (HMul.hMul (HMul.hMul (HPow.hPow ( …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (b i) (AkraBazziRecurrence.p a b)) (H …
  -/
  show (b i) ^ (p a b) * n ^ (p a b) * (1 + ε n) ≤ q (r i n)
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q (HMul.hMul (b i) ↑n) …
    n : Nat
    hn : LE.le (HSub.hSub (q (HMul.hMul (b i) ↑n)) (q ↑(r i n))) (HMul.hMul (HMul. …
    h₁ : Eq (HSub.hSub (q (HMul.hMul (b i) ↑n)) (HMul.hMul (HMul.hMul (HPow.hPow ( …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (b i) (AkraBazziRecurrence.p a b)) (H …
  -/
  rw [← h₁, sub_le_iff_le_add', ← sub_le_iff_le_add]
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    i : α
    q : Real → Real := fun x => HMul.hMul (HPow.hPow x (AkraBazziRecurrence.p a b) …
    h_diff_q : DifferentiableOn Real q (Set.Ioi 1)
    h_deriv_q : Asymptotics.IsBigO Filter.atTop (deriv q) fun x => HPow.hPow x (HS …
    h_main_norm : Filter.atTop.EventuallyLE (fun n => Norm.norm (HSub.hSub (q ↑(r  …
    h_main : Filter.atTop.EventuallyLE (fun n => HSub.hSub (q (HMul.hMul (b i) ↑n) …
    n : Nat
    hn : LE.le (HSub.hSub (q (HMul.hMul (b i) ↑n)) (q ↑(r i n))) (HMul.hMul (HMul. …
    h₁ : Eq (HSub.hSub (q (HMul.hMul (b i) ↑n)) (HMul.hMul (HMul.hMul (HPow.hPow ( …
    ⊢ LE.le (HSub.hSub (q (HMul.hMul (b i) ↑n)) (q ↑(r i n))) (HMul.hMul (HMul.hMu …
  -/
  exact hn
  /-
    🎉 no goals
  -/


lemma base_nonempty {n : ℕ} (hn : 0 < n) : (Finset.Ico (⌊b (min_bi b) / 2 * n⌋₊) n).Nonempty := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : LT.lt 0 n
    ⊢ (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRecurrence.min_bi  …
  -/
  let b' := b (min_bi b)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : LT.lt 0 n
    b' : Real := b (AkraBazziRecurrence.min_bi b)
    ⊢ (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRecurrence.min_bi  …
  -/
  have hb_pos : 0 < b' := R.b_pos _
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    n : Nat
    hn : LT.lt 0 n
    b' : Real := b (AkraBazziRecurrence.min_bi b)
    hb_pos : LT.lt 0 b'
    ⊢ (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRecurrence.min_bi  …
  -/
  simp_rw [Finset.nonempty_Ico]
  exact_mod_cast calc ⌊b' / 2 * n⌋₊ ≤ b' / 2 * n := by exact Nat.floor_le (by positivity)
                                 _ < 1 / 2 * n   := by gcongr; exact R.b_lt_one (min_bi b)
                                 _ ≤ 1 * n       := by gcongr; norm_num
                                 _ = n           := by simp


/-- The main proof of the upper bound part of the Akra-Bazzi theorem. The factor
`1 - ε n` does not change the asymptotic order, but is needed for the induction step to go
through. -/
lemma T_isBigO_smoothingFn_mul_asympBound :
    T =O[atTop] (fun n => (1 - ε n) * asympBound g a b n) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Asymptotics.IsBigO Filter.atTop T fun n => HMul.hMul (HSub.hSub 1 (AkraBazzi …
  -/
  let b' := b (min_bi b) / 2
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    ⊢ Asymptotics.IsBigO Filter.atTop T fun n => HMul.hMul (HSub.hSub 1 (AkraBazzi …
  -/
  have hb_pos : 0 < b' := R.bi_min_div_two_pos
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    ⊢ Asymptotics.IsBigO Filter.atTop T fun n => HMul.hMul (HSub.hSub 1 (AkraBazzi …
  -/
  rw [isBigO_atTop_iff_eventually_exists]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    ⊢ Filter.Eventually (fun n₀ => Exists fun c => ∀ (n : Nat), GE.ge n n₀ → LE.le …
  -/
  obtain ⟨c₁, hc₁, h_sumTransform_aux⟩ := R.eventually_atTop_sumTransform_ge
  filter_upwards [eventually_ge_atTop R.n₀,       -- n₀_ge_Rn₀
      eventually_forall_ge_atTop.mpr eventually_one_sub_smoothingFn_pos,    -- h_smoothing_pos
      eventually_forall_ge_atTop.mpr
        <| eventually_one_sub_smoothingFn_gt_const (1/2) (by norm_num),    -- h_smoothing_gt_half
      eventually_forall_ge_atTop.mpr R.eventually_asympBound_pos,            -- h_asympBound_pos
      eventually_forall_ge_atTop.mpr R.eventually_asympBound_r_pos,          -- h_asympBound_r_pos
      (tendsto_nat_floor_mul_atTop b' hb_pos).eventually_forall_ge_atTop
        R.eventually_asympBound_pos,   -- h_asympBound_floor
      eventually_gt_atTop 0,                                                -- n₀_pos
      eventually_forall_ge_atTop.mpr R.eventually_one_sub_smoothingFn_r_pos,  -- h_smoothing_r_pos
      eventually_forall_ge_atTop.mpr R.rpow_p_mul_one_sub_smoothingFn_le,    -- bound1
      (tendsto_nat_floor_mul_atTop b' hb_pos).eventually_forall_ge_atTop
        eventually_one_sub_smoothingFn_pos,   -- h_smoothingFn_floor
      eventually_forall_ge_atTop.mpr h_sumTransform_aux,                     -- h_sumTransform
      eventually_forall_ge_atTop.mpr R.eventually_bi_mul_le_r]               -- h_bi_le_r
    with n₀ n₀_ge_Rn₀ h_smoothing_pos h_smoothing_gt_half
      h_asympBound_pos h_asympBound_r_pos h_asympBound_floor n₀_pos h_smoothing_r_pos
      bound1 h_smoothingFn_floor h_sumTransform h_bi_le_r
  -- Max of the ratio `T(n) / asympBound(n)` over the base case `n ∈ [b * n₀, n₀)`
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HSub.hSub 1 (AkraBazziRec …
    h_smoothing_gt_half : ∀ (y : Nat), LE.le n₀ y → LT.lt (1 / 2) (HSub.hSub 1 (Ak …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HSub.hSub 1  …
    bound1 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul c₁ (g ↑ …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    ⊢ Exists fun c => ∀ (n : Nat), GE.ge n n₀ → LE.le (Norm.norm (T n)) (HMul.hMul …
  -/
  have h_base_nonempty := R.base_nonempty n₀_pos
  let base_max : ℝ :=
    (Finset.Ico (⌊b' * n₀⌋₊) n₀).sup' h_base_nonempty
      fun n => T n / ((1 - ε n) * asympBound g a b n)
  -- The big-O constant we are aiming for: max of the base case ratio and what we need to
  -- cancel out the `g(n)` term in the calculation below
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HSub.hSub 1 (AkraBazziRec …
    h_smoothing_gt_half : ∀ (y : Nat), LE.le n₀ y → LT.lt (1 / 2) (HSub.hSub 1 (Ak …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HSub.hSub 1  …
    bound1 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul c₁ (g ↑ …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_max : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).sup' h_base_ …
    ⊢ Exists fun c => ∀ (n : Nat), GE.ge n n₀ → LE.le (Norm.norm (T n)) (HMul.hMul …
  -/
  set C := max (2 * c₁⁻¹) base_max with hC
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HSub.hSub 1 (AkraBazziRec …
    h_smoothing_gt_half : ∀ (y : Nat), LE.le n₀ y → LT.lt (1 / 2) (HSub.hSub 1 (Ak …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HSub.hSub 1  …
    bound1 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul c₁ (g ↑ …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_max : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).sup' h_base_ …
    C : Real := Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max
    hC : Eq C (Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max)
    ⊢ Exists fun c => ∀ (n : Nat), GE.ge n n₀ → LE.le (Norm.norm (T n)) (HMul.hMul …
  -/
  refine ⟨C, fun n hn => ?_⟩
  -- Base case: statement is true for `b' * n₀ ≤ n < n₀`
  have h_base : ∀ n ∈ Finset.Ico (⌊b' * n₀⌋₊) n₀, T n ≤ C * ((1 - ε n) * asympBound g a b n) := by
    intro n hn
    rw [Finset.mem_Ico] at hn
    have htmp1 : 0 < 1 - ε n := h_smoothingFn_floor n hn.1
    have htmp2 : 0 < asympBound g a b n := h_asympBound_floor n hn.1
    rw [← _root_.div_le_iff₀ (by positivity)]
    rw [← Finset.mem_Ico] at hn
    calc T n / ((1 - ε ↑n) * asympBound g a b n)
           ≤ (Finset.Ico (⌊b' * n₀⌋₊) n₀).sup' h_base_nonempty
                (fun z => T z / ((1 - ε z) * asympBound g a b z)) :=
                  Finset.le_sup'_of_le _ (b := n) hn le_rfl
         _ ≤ C := le_max_right _ _
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HSub.hSub 1 (AkraBazziRec …
    h_smoothing_gt_half : ∀ (y : Nat), LE.le n₀ y → LT.lt (1 / 2) (HSub.hSub 1 (Ak …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HSub.hSub 1  …
    bound1 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul c₁ (g ↑ …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_max : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).sup' h_base_ …
    C : Real := Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max
    hC : Eq C (Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max)
    n : Nat
    hn : GE.ge n n₀
    h_base : ∀ (n : Nat), Membership.mem (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀) …
    ⊢ LE.le (Norm.norm (T n)) (HMul.hMul C (Norm.norm (HMul.hMul (HSub.hSub 1 (Akr …
  -/
  have h_asympBound_pos' : 0 < asympBound g a b n := h_asympBound_pos n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HSub.hSub 1 (AkraBazziRec …
    h_smoothing_gt_half : ∀ (y : Nat), LE.le n₀ y → LT.lt (1 / 2) (HSub.hSub 1 (Ak …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HSub.hSub 1  …
    bound1 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul c₁ (g ↑ …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_max : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).sup' h_base_ …
    C : Real := Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max
    hC : Eq C (Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max)
    n : Nat
    hn : GE.ge n n₀
    h_base : ∀ (n : Nat), Membership.mem (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀) …
    h_asympBound_pos' : LT.lt 0 (AkraBazziRecurrence.asympBound g a b n)
    ⊢ LE.le (Norm.norm (T n)) (HMul.hMul C (Norm.norm (HMul.hMul (HSub.hSub 1 (Akr …
  -/
  have h_one_sub_smoothingFn_pos' : 0 < 1 - ε n := h_smoothing_pos n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (HMul.hMul c …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HSub.hSub 1 (AkraBazziRec …
    h_smoothing_gt_half : ∀ (y : Nat), LE.le n₀ y → LT.lt (1 / 2) (HSub.hSub 1 (Ak …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HSub.hSub 1  …
    bound1 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HPow.hPow (↑(r …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul c₁ (g ↑ …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_max : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).sup' h_base_ …
    C : Real := Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max
    hC : Eq C (Max.max (HMul.hMul 2 (Inv.inv c₁)) base_max)
    n : Nat
    hn : GE.ge n n₀
    h_base : ∀ (n : Nat), Membership.mem (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀) …
    h_asympBound_pos' : LT.lt 0 (AkraBazziRecurrence.asympBound g a b n)
    h_one_sub_smoothingFn_pos' : LT.lt 0 (HSub.hSub 1 (AkraBazziRecurrence.smoothi …
    ⊢ LE.le (Norm.norm (T n)) (HMul.hMul C (Norm.norm (HMul.hMul (HSub.hSub 1 (Akr …
  -/
  rw [Real.norm_of_nonneg (R.T_nonneg n), Real.norm_of_nonneg (by positivity)]
  -- We now prove all other cases by induction
  induction n using Nat.strongRecOn with
  | ind n h_ind =>
    have b_mul_n₀_le_ri i : ⌊b' * ↑n₀⌋₊ ≤ r i n := by
      exact_mod_cast calc ⌊b' * (n₀ : ℝ)⌋₊ ≤ b' * n₀ := Nat.floor_le <| by positivity
                                  _ ≤ b' * n         := by gcongr
                                  _ ≤ r i n          := h_bi_le_r n hn i
    have g_pos : 0 ≤ g n := R.g_nonneg n (by positivity)
    calc
      T n = (∑ i, a i * T (r i n)) + g n := by exact R.h_rec n <| n₀_ge_Rn₀.trans hn
        _ ≤ (∑ i, a i * (C * ((1 - ε (r i n)) * asympBound g a b (r i n)))) + g n := by
            -- Apply the induction hypothesis, or use the base case depending on how large n is
            gcongr (∑ i, a i * ?_) + g n with i _
            · exact le_of_lt <| R.a_pos _
            · if ri_lt_n₀ : r i n < n₀ then
                exact h_base _ <| by
                  simp_all only [gt_iff_lt, Nat.ofNat_pos, div_pos_iff_of_pos_right,
                    eventually_atTop, sub_pos, one_div, mem_Ico, and_imp,
                    forall_true_left, mem_univ, and_self, b', C, base_max]
              else
                push_neg at ri_lt_n₀
                exact h_ind (r i n) (R.r_lt_n _ _ (n₀_ge_Rn₀.trans hn)) ri_lt_n₀
                  (h_asympBound_r_pos _ hn _) (h_smoothing_r_pos n hn i)
        _ = (∑ i, a i * (C * ((1 - ε (r i n)) * ((r i n) ^ (p a b)
                * (1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1))))))) + g n := by
            simp_rw [asympBound_def']
        _ = (∑ i, C * a i * ((r i n) ^ (p a b) * (1 - ε (r i n))
                * ((1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1)))))) + g n := by
            congr; ext; ring
        _ ≤ (∑ i, C * a i * ((b i) ^ (p a b) * n ^ (p a b) * (1 - ε n)
                * ((1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1)))))) + g n := by
            gcongr (∑ i, C * a i * (?_
                * ((1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1)))))) + g n with i
            · have := R.a_pos i
              positivity
            · refine add_nonneg zero_le_one <| Finset.sum_nonneg fun j _ => ?_
              rw [div_nonneg_iff]
              exact Or.inl ⟨R.g_nonneg j (by positivity), by positivity⟩
            · exact bound1 n hn i
        _ = (∑ i, C * a i * ((b i) ^ (p a b) * n ^ (p a b) * (1 - ε n)
                * ((1 + ((∑ u ∈ range n, g u / u ^ ((p a b) + 1))
                - (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1))))))) + g n := by
            congr; ext i; congr
            refine eq_sub_of_add_eq ?_
            rw [add_comm]
            exact add_eq_of_eq_sub <| Finset.sum_Ico_eq_sub _
              <| le_of_lt <| R.r_lt_n i n <| n₀_ge_Rn₀.trans hn
        _ = (∑ i, C * a i * ((b i) ^ (p a b) * (1 - ε n) * ((n ^ (p a b)
                * (1 + (∑ u ∈ range n, g u / u ^ ((p a b) + 1)))
                - n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1))))))
                + g n := by
            congr; ext; ring
        _ = (∑ i, C * a i * ((b i) ^ (p a b) * (1 - ε n)
                * ((asympBound g a b n - sumTransform (p a b) g (r i n) n)))) + g n := by
            simp_rw [asympBound_def', sumTransform_def]
        _ ≤ (∑ i, C * a i * ((b i) ^ (p a b) * (1 - ε n)
                * ((asympBound g a b n - c₁ * g n)))) + g n := by
            gcongr with i
            · have := R.a_pos i
              positivity
            · have := R.b_pos i
              positivity
            · exact h_sumTransform n hn i
        _ = (∑ i, C * (1 - ε n) * ((asympBound g a b n - c₁ * g n))
                * (a i * (b i) ^ (p a b))) + g n := by
            congr; ext; ring
        _ = C * (1 - ε n) * (asympBound g a b n - c₁ * g n) + g n := by
            rw [← Finset.mul_sum, R.sumCoeffsExp_p_eq_one, mul_one]
        _ = C * (1 - ε n) * asympBound g a b n + (1 - C * c₁ * (1 - ε n)) * g n := by ring
        _ ≤ C * (1 - ε n) * asympBound g a b n + 0 := by
            gcongr
            refine mul_nonpos_of_nonpos_of_nonneg ?_ g_pos
            rw [sub_nonpos]
            calc 1 ≤ 2 * (c₁⁻¹ * c₁) * (1/2) := by
                    rw [inv_mul_cancel₀ (by positivity : c₁ ≠ 0)]; norm_num
                 _ = (2 * c₁⁻¹) * c₁ * (1/2) := by ring
                 _ ≤ C * c₁ * (1 - ε n) := by gcongr
                                              · rw [hC]; exact le_max_left _ _
                                              · exact le_of_lt <| h_smoothing_gt_half n hn
        _ = C * ((1 - ε n) * asympBound g a b n) := by ring


/-- The main proof of the lower bound part of the Akra-Bazzi theorem. The factor
`1 + ε n` does not change the asymptotic order, but is needed for the induction step to go
through. -/
lemma smoothingFn_mul_asympBound_isBigO_T :
    (fun (n : ℕ) => (1 + ε n) * asympBound g a b n) =O[atTop] T := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HAdd.hAdd 1 (AkraBazziR …
  -/
  let b' := b (min_bi b) / 2
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HAdd.hAdd 1 (AkraBazziR …
  -/
  have hb_pos : 0 < b' := R.bi_min_div_two_pos
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HAdd.hAdd 1 (AkraBazziR …
  -/
  rw [isBigO_atTop_iff_eventually_exists_pos]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    ⊢ Filter.Eventually (fun n₀ => Exists fun c => And (GT.gt c 0) (∀ (n : Nat), G …
  -/
  obtain ⟨c₁, hc₁, h_sumTransform_aux⟩ := R.eventually_atTop_sumTransform_le
  filter_upwards [eventually_ge_atTop R.n₀,                                 -- n₀_ge_Rn₀
      (tendsto_nat_floor_mul_atTop b' hb_pos).eventually_gt_atTop 0,        -- h_b_floor
      eventually_forall_ge_atTop.mpr eventually_one_add_smoothingFn_pos,    -- h_smoothing_pos
      (tendsto_nat_floor_mul_atTop b' hb_pos).eventually_forall_ge_atTop
        eventually_one_add_smoothingFn_pos,                                 -- h_smoothing_pos'
      eventually_forall_ge_atTop.mpr R.eventually_asympBound_pos,            -- h_asympBound_pos
      eventually_forall_ge_atTop.mpr R.eventually_asympBound_r_pos,          -- h_asympBound_r_pos
      (tendsto_nat_floor_mul_atTop b' hb_pos).eventually_forall_ge_atTop
        R.eventually_asympBound_pos,                                         -- h_asympBound_floor
      eventually_gt_atTop 0,                                                -- n₀_pos
      eventually_forall_ge_atTop.mpr R.eventually_one_add_smoothingFn_r_pos,  -- h_smoothing_r_pos
      eventually_forall_ge_atTop.mpr R.rpow_p_mul_one_add_smoothingFn_ge,   -- bound2
      (tendsto_nat_floor_mul_atTop b' hb_pos).eventually_forall_ge_atTop
        eventually_one_add_smoothingFn_pos,                                 -- h_smoothingFn_floor
      eventually_forall_ge_atTop.mpr h_sumTransform_aux,                    -- h_sumTransform
      eventually_forall_ge_atTop.mpr R.eventually_bi_mul_le_r,              -- h_bi_le_r
      eventually_forall_ge_atTop.mpr (eventually_ge_atTop ⌈exp 1⌉₊)]        -- h_exp
    with n₀ n₀_ge_Rn₀ h_b_floor h_smoothing_pos h_smoothing_pos' h_asympBound_pos h_asympBound_r_pos
      h_asympBound_floor n₀_pos h_smoothing_r_pos bound2 h_smoothingFn_floor h_sumTransform
      h_bi_le_r h_exp
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (AkraBazziRe …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_b_floor : LT.lt 0 (Nat.floor (HMul.hMul b' ↑n₀))
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HAdd.hAdd 1 (AkraBazziRec …
    h_smoothing_pos' : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT.lt …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HAdd.hAdd 1  …
    bound2 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPo …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (AkraBazziRecurren …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_exp : ∀ (y : Nat), LE.le n₀ y → LE.le (Nat.ceil (Real.exp 1)) y
    ⊢ Exists fun c => And (GT.gt c 0) (∀ (n : Nat), GE.ge n n₀ → LE.le (HMul.hMul  …
  -/
  have h_base_nonempty := R.base_nonempty n₀_pos
  -- Min of the ratio T(n) / asympBound(n) over the base case n ∈ [b * n₀, n₀)
  set base_min : ℝ :=
    (Finset.Ico (⌊b' * n₀⌋₊) n₀).inf' h_base_nonempty
      (fun n => T n / ((1 + ε n) * asympBound g a b n)) with base_min_def
  -- The big-O constant we are aiming for: min of the base case ratio and what we need to cancel
  -- out the g(n) term in the calculation below
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (AkraBazziRe …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_b_floor : LT.lt 0 (Nat.floor (HMul.hMul b' ↑n₀))
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HAdd.hAdd 1 (AkraBazziRec …
    h_smoothing_pos' : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT.lt …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HAdd.hAdd 1  …
    bound2 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPo …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (AkraBazziRecurren …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_exp : ∀ (y : Nat), LE.le n₀ y → LE.le (Nat.ceil (Real.exp 1)) y
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_min : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf' h_base_ …
    base_min_def : Eq base_min ((Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf …
    ⊢ Exists fun c => And (GT.gt c 0) (∀ (n : Nat), GE.ge n n₀ → LE.le (HMul.hMul  …
  -/
  let C := min (2 * c₁)⁻¹ base_min
  have hC_pos : 0 < C := by
    refine lt_min (by positivity) ?_
    obtain ⟨m, hm_mem, hm⟩ :=
      Finset.exists_mem_eq_inf' h_base_nonempty (fun n => T n / ((1 + ε n) * asympBound g a b n))
    calc 0 < T m / ((1 + ε m) * asympBound g a b m) := by
              have H₁ : 0 < T m := by exact R.T_pos _
              have H₂ : 0 < 1 + ε m := by rw [Finset.mem_Ico] at hm_mem
                                          exact h_smoothing_pos' m hm_mem.1
              have H₃ : 0 < asympBound g a b m := by
                refine R.asympBound_pos m ?_
                calc 0 < ⌊b' * n₀⌋₊ := by exact h_b_floor
                     _ ≤ m := by rw [Finset.mem_Ico] at hm_mem; exact hm_mem.1
              positivity
         _ = base_min := by rw [base_min_def, hm]
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (AkraBazziRe …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_b_floor : LT.lt 0 (Nat.floor (HMul.hMul b' ↑n₀))
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HAdd.hAdd 1 (AkraBazziRec …
    h_smoothing_pos' : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT.lt …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HAdd.hAdd 1  …
    bound2 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPo …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (AkraBazziRecurren …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_exp : ∀ (y : Nat), LE.le n₀ y → LE.le (Nat.ceil (Real.exp 1)) y
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_min : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf' h_base_ …
    base_min_def : Eq base_min ((Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf …
    C : Real := Min.min (Inv.inv (HMul.hMul 2 c₁)) base_min
    hC_pos : LT.lt 0 C
    ⊢ Exists fun c => And (GT.gt c 0) (∀ (n : Nat), GE.ge n n₀ → LE.le (HMul.hMul  …
  -/
  refine ⟨C, hC_pos, fun n hn => ?_⟩
  -- Base case: statement is true for `b' * n₀ ≤ n < n₀`
  have h_base : ∀ n ∈ Finset.Ico (⌊b' * n₀⌋₊) n₀, C * ((1 + ε n) * asympBound g a b n) ≤ T n := by
    intro n hn
    rw [Finset.mem_Ico] at hn
    have htmp1 : 0 < 1 + ε n := h_smoothingFn_floor n hn.1
    have htmp2 : 0 < asympBound g a b n := h_asympBound_floor n hn.1
    rw [← _root_.le_div_iff₀ (by positivity)]
    rw [← Finset.mem_Ico] at hn
    calc T n / ((1 + ε ↑n) * asympBound g a b n)
           ≥ (Finset.Ico (⌊b' * n₀⌋₊) n₀).inf' h_base_nonempty
                  fun z => T z / ((1 + ε z) * asympBound g a b z) :=
                    Finset.inf'_le_of_le _ (b := n) hn <| le_refl _
         _ ≥ C := min_le_right _ _
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (AkraBazziRe …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_b_floor : LT.lt 0 (Nat.floor (HMul.hMul b' ↑n₀))
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HAdd.hAdd 1 (AkraBazziRec …
    h_smoothing_pos' : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT.lt …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HAdd.hAdd 1  …
    bound2 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPo …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (AkraBazziRecurren …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_exp : ∀ (y : Nat), LE.le n₀ y → LE.le (Nat.ceil (Real.exp 1)) y
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_min : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf' h_base_ …
    base_min_def : Eq base_min ((Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf …
    C : Real := Min.min (Inv.inv (HMul.hMul 2 c₁)) base_min
    hC_pos : LT.lt 0 C
    n : Nat
    hn : GE.ge n n₀
    h_base : ∀ (n : Nat), Membership.mem (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀) …
    ⊢ LE.le (HMul.hMul C (Norm.norm (HMul.hMul (HAdd.hAdd 1 (AkraBazziRecurrence.s …
  -/
  have h_asympBound_pos' : 0 < asympBound g a b n := h_asympBound_pos n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (AkraBazziRe …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_b_floor : LT.lt 0 (Nat.floor (HMul.hMul b' ↑n₀))
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HAdd.hAdd 1 (AkraBazziRec …
    h_smoothing_pos' : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT.lt …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HAdd.hAdd 1  …
    bound2 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPo …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (AkraBazziRecurren …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_exp : ∀ (y : Nat), LE.le n₀ y → LE.le (Nat.ceil (Real.exp 1)) y
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_min : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf' h_base_ …
    base_min_def : Eq base_min ((Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf …
    C : Real := Min.min (Inv.inv (HMul.hMul 2 c₁)) base_min
    hC_pos : LT.lt 0 C
    n : Nat
    hn : GE.ge n n₀
    h_base : ∀ (n : Nat), Membership.mem (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀) …
    h_asympBound_pos' : LT.lt 0 (AkraBazziRecurrence.asympBound g a b n)
    ⊢ LE.le (HMul.hMul C (Norm.norm (HMul.hMul (HAdd.hAdd 1 (AkraBazziRecurrence.s …
  -/
  have h_one_sub_smoothingFn_pos' : 0 < 1 + ε n := h_smoothing_pos n hn
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    T : Nat → Real
    g : Real → Real
    a b : α → Real
    r : α → Nat → Nat
    inst✝ : Nonempty α
    R : AkraBazziRecurrence T g a b r
    b' : Real := HDiv.hDiv (b (AkraBazziRecurrence.min_bi b)) 2
    hb_pos : LT.lt 0 b'
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    h_sumTransform_aux : Filter.Eventually (fun n => ∀ (i : α), LE.le (AkraBazziRe …
    n₀ : Nat
    n₀_ge_Rn₀ : LE.le R.n₀ n₀
    h_b_floor : LT.lt 0 (Nat.floor (HMul.hMul b' ↑n₀))
    h_smoothing_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (HAdd.hAdd 1 (AkraBazziRec …
    h_smoothing_pos' : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT.lt …
    h_asympBound_pos : ∀ (y : Nat), LE.le n₀ y → LT.lt 0 (AkraBazziRecurrence.asym …
    h_asympBound_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (AkraBazziRe …
    h_asympBound_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT. …
    n₀_pos : LT.lt 0 n₀
    h_smoothing_r_pos : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LT.lt 0 (HAdd.hAdd 1  …
    bound2 : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HMul.hMul (HPo …
    h_smoothingFn_floor : ∀ (y : Nat), LE.le (Nat.floor (HMul.hMul b' ↑n₀)) y → LT …
    h_sumTransform : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (AkraBazziRecurren …
    h_bi_le_r : ∀ (y : Nat), LE.le n₀ y → ∀ (i : α), LE.le (HMul.hMul (HDiv.hDiv ( …
    h_exp : ∀ (y : Nat), LE.le n₀ y → LE.le (Nat.ceil (Real.exp 1)) y
    h_base_nonempty : (Finset.Ico (Nat.floor (HMul.hMul (HDiv.hDiv (b (AkraBazziRe …
    base_min : Real := (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf' h_base_ …
    base_min_def : Eq base_min ((Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀)) n₀).inf …
    C : Real := Min.min (Inv.inv (HMul.hMul 2 c₁)) base_min
    hC_pos : LT.lt 0 C
    n : Nat
    hn : GE.ge n n₀
    h_base : ∀ (n : Nat), Membership.mem (Finset.Ico (Nat.floor (HMul.hMul b' ↑n₀) …
    h_asympBound_pos' : LT.lt 0 (AkraBazziRecurrence.asympBound g a b n)
    h_one_sub_smoothingFn_pos' : LT.lt 0 (HAdd.hAdd 1 (AkraBazziRecurrence.smoothi …
    ⊢ LE.le (HMul.hMul C (Norm.norm (HMul.hMul (HAdd.hAdd 1 (AkraBazziRecurrence.s …
  -/
  rw [Real.norm_of_nonneg (R.T_nonneg n), Real.norm_of_nonneg (by positivity)]
  -- We now prove all other cases by induction
  induction n using Nat.strongRecOn with
  | ind n h_ind =>
    have b_mul_n₀_le_ri i : ⌊b' * ↑n₀⌋₊ ≤ r i n := by
      exact_mod_cast calc ⌊b' * ↑n₀⌋₊ ≤ b' * n₀ := Nat.floor_le <| by positivity
                                  _ ≤ b' * n := by gcongr
                                  _ ≤ r i n := h_bi_le_r n hn i
    have g_pos : 0 ≤ g n := R.g_nonneg n (by positivity)
    calc
      T n = (∑ i, a i * T (r i n)) + g n := by exact R.h_rec n <| n₀_ge_Rn₀.trans hn
        _ ≥ (∑ i, a i * (C * ((1 + ε (r i n)) * asympBound g a b (r i n)))) + g n := by
            -- Apply the induction hypothesis, or use the base case depending on how large `n` is
              gcongr (∑ i, a i * ?_) + g n with i _
              · exact le_of_lt <| R.a_pos _
              · cases lt_or_le (r i n) n₀ with
                | inl ri_lt_n₀ => exact h_base _ <| Finset.mem_Ico.mpr ⟨b_mul_n₀_le_ri i, ri_lt_n₀⟩
                | inr n₀_le_ri =>
                  exact h_ind (r i n) (R.r_lt_n _ _ (n₀_ge_Rn₀.trans hn)) n₀_le_ri
                    (h_asympBound_r_pos _ hn _) (h_smoothing_r_pos n hn i)
        _ = (∑ i, a i * (C * ((1 + ε (r i n)) * ((r i n) ^ (p a b)
                  * (1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1))))))) + g n := by
              simp_rw [asympBound_def']
        _ = (∑ i, C * a i * ((r i n)^(p a b) * (1 + ε (r i n))
                  * ((1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1)))))) + g n := by
              congr; ext; ring
        _ ≥ (∑ i, C * a i * ((b i) ^ (p a b) * n ^ (p a b) * (1 + ε n)
                  * ((1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1)))))) + g n := by
              gcongr (∑ i, C * a i * (?_ *
                  ((1 + (∑ u ∈ range (r i n), g u / u ^ ((p a b) + 1)))))) + g n with i
              · have := R.a_pos i
                positivity
              · refine add_nonneg zero_le_one <| Finset.sum_nonneg fun j _ => ?_
                rw [div_nonneg_iff]
                exact Or.inl ⟨R.g_nonneg j (by positivity), by positivity⟩
              · exact bound2 n hn i
        _ = (∑ i, C * a i * ((b i) ^ (p a b) * n ^ (p a b) * (1 + ε n)
                  * ((1 + ((∑ u ∈ range n, g u / u ^ ((p a b) + 1))
                  - (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1))))))) + g n := by
              congr; ext i; congr
              refine eq_sub_of_add_eq ?_
              rw [add_comm]
              exact add_eq_of_eq_sub <| Finset.sum_Ico_eq_sub _
                <| le_of_lt <| R.r_lt_n i n <| n₀_ge_Rn₀.trans hn
        _ = (∑ i, C * a i * ((b i) ^ (p a b) * (1 + ε n)
                  * ((n ^ (p a b) * (1 + (∑ u ∈ range n, g u / u ^ ((p a b) + 1)))
                  - n ^ (p a b) * (∑ u ∈ Finset.Ico (r i n) n, g u / u ^ ((p a b) + 1))))))
                  + g n := by
              congr; ext; ring
        _ = (∑ i, C * a i * ((b i) ^ (p a b) * (1 + ε n)
                  * ((asympBound g a b n - sumTransform (p a b) g (r i n) n)))) + g n := by
              simp_rw [asympBound_def', sumTransform_def]
        _ ≥ (∑ i, C * a i * ((b i) ^ (p a b) * (1 + ε n)
                  * ((asympBound g a b n - c₁ * g n)))) + g n := by
              gcongr with i
              · have := R.a_pos i
                positivity
              · have := R.b_pos i
                positivity
              · exact h_sumTransform n hn i
        _ = (∑ i, C * (1 + ε n) * ((asympBound g a b n - c₁ * g n))
                  * (a i * (b i) ^ (p a b))) + g n := by congr; ext; ring
        _ = C * (1 + ε n) * (asympBound g a b n - c₁ * g n) + g n := by
              rw [← Finset.mul_sum, R.sumCoeffsExp_p_eq_one, mul_one]
        _ = C * (1 + ε n) * asympBound g a b n + (1 - C * c₁ * (1 + ε n)) * g n := by ring
        _ ≥ C * (1 + ε n) * asympBound g a b n + 0 := by
              gcongr
              refine mul_nonneg ?_ g_pos
              rw [sub_nonneg]
              calc C * c₁ * (1 + ε n) ≤ C * c₁ * 2 := by
                        gcongr
                        refine one_add_smoothingFn_le_two ?_
                        calc exp 1 ≤ ⌈exp 1⌉₊ := by exact Nat.le_ceil _
                                 _ ≤ n := by exact_mod_cast h_exp n hn
                    _ = C * (2 * c₁) := by ring
                    _ ≤ (2 * c₁)⁻¹ * (2 * c₁) := by gcongr; exact min_le_left _ _
                    _ = c₁⁻¹ * c₁ := by ring
                    _ = 1 := inv_mul_cancel₀ (by positivity)
        _ = C * ((1 + ε n) * asympBound g a b n) := by ring


/-- The **Akra-Bazzi theorem**: `T ∈ O(n^p (1 + ∑_u^n g(u) / u^{p+1}))` -/
theorem isBigO_asympBound : T =O[atTop] asympBound g a b := by
  calc T =O[atTop] (fun n => (1 - ε n) * asympBound g a b n) := by
              exact R.T_isBigO_smoothingFn_mul_asympBound
         _ =O[atTop] (fun n => 1 * asympBound g a b n) := by
              refine IsBigO.mul (isBigO_const_of_tendsto (y := 1) ?_ one_ne_zero)
                (isBigO_refl _ _)
              rw [← Function.comp_def (fun n => 1 - ε n) Nat.cast]
              exact Tendsto.comp isEquivalent_one_sub_smoothingFn_one.tendsto_const
                tendsto_natCast_atTop_atTop
         _ = asympBound g a b := by simp


/-- The **Akra-Bazzi theorem**: `T ∈ Ω(n^p (1 + ∑_u^n g(u) / u^{p+1}))` -/
theorem isBigO_symm_asympBound : asympBound g a b =O[atTop] T := by
  calc asympBound g a b = (fun n => 1 * asympBound g a b n) := by simp
                 _ ~[atTop] (fun n => (1 + ε n) * asympBound g a b n) := by
                            refine IsEquivalent.mul (IsEquivalent.symm ?_) IsEquivalent.refl
                            rw [Function.const_def, isEquivalent_const_iff_tendsto one_ne_zero,
                              ← Function.comp_def (fun n => 1 + ε n) Nat.cast]
                            exact Tendsto.comp isEquivalent_one_add_smoothingFn_one.tendsto_const
                              tendsto_natCast_atTop_atTop
                 _ =O[atTop] T := R.smoothingFn_mul_asympBound_isBigO_T


/-- The **Akra-Bazzi theorem**: `T ∈ Θ(n^p (1 + ∑_u^n g(u) / u^{p+1}))` -/
theorem isTheta_asympBound : T =Θ[atTop] asympBound g a b :=
  ⟨R.isBigO_asympBound, R.isBigO_symm_asympBound⟩


