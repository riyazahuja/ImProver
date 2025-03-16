/-- A real-valued sequence is subadditive if it satisfies the inequality `u (m + n) ≤ u m + u n`
for all `m, n`. -/
def Subadditive (u : ℕ → ℝ) : Prop :=
  ∀ m n, u (m + n) ≤ u m + u n


/-- The limit of a bounded-below subadditive sequence. The fact that the sequence indeed tends to
this limit is given in `Subadditive.tendsto_lim` -/
@[nolint unusedArguments] -- Porting note: was irreducible
protected def lim (_h : Subadditive u) :=
  sInf ((fun n : ℕ => u n / n) '' Ici 1)


theorem lim_le_div (hbdd : BddBelow (range fun n => u n / n)) {n : ℕ} (hn : n ≠ 0) :
    h.lim ≤ u n / n := by
  /-
    u : Nat → Real
    h : Subadditive u
    hbdd : BddBelow (Set.range fun n => HDiv.hDiv (u n) ↑n)
    n : Nat
    hn : Ne n 0
    ⊢ LE.le h.lim (HDiv.hDiv (u n) ↑n)
  -/
  rw [Subadditive.lim]
  /-
    u : Nat → Real
    h : Subadditive u
    hbdd : BddBelow (Set.range fun n => HDiv.hDiv (u n) ↑n)
    n : Nat
    hn : Ne n 0
    ⊢ LE.le (InfSet.sInf (Set.image (fun n => HDiv.hDiv (u n) ↑n) (Set.Ici 1))) (H …
  -/
  exact csInf_le (hbdd.mono <| image_subset_range _ _) ⟨n, hn.bot_lt, rfl⟩
  /-
    🎉 no goals
  -/


include h in
theorem apply_mul_add_le (k n r) : u (k * n + r) ≤ k * u n + u r := by
  induction k with
  | zero => simp only [Nat.cast_zero, zero_mul, zero_add]; rfl
  | succ k IH =>
    calc
      u ((k + 1) * n + r) = u (n + (k * n + r)) := by congr 1; ring
      _ ≤ u n + u (k * n + r) := h _ _
      _ ≤ u n + (k * u n + u r) := add_le_add_left IH _
      _ = (k + 1 : ℕ) * u n + u r := by simp; ring


include h in
theorem eventually_div_lt_of_div_lt {L : ℝ} {n : ℕ} (hn : n ≠ 0) (hL : u n / n < L) :
    ∀ᶠ p in atTop, u p / p < L := by
  /- It suffices to prove the statement for each arithmetic progression `(n * · + r)`. -/
  /-
    u : Nat → Real
    h : Subadditive u
    L : Real
    n : Nat
    hn : Ne n 0
    hL : LT.lt (HDiv.hDiv (u n) ↑n) L
    ⊢ Filter.Eventually (fun p => LT.lt (HDiv.hDiv (u p) ↑p) L) Filter.atTop
  -/
  refine .atTop_of_arithmetic hn fun r _ => ?_
  /- `(k * u n + u r) / (k * n + r)` tends to `u n / n < L`, hence
  `(k * u n + u r) / (k * n + r) < L` for sufficiently large `k`. -/
  have A : Tendsto (fun x : ℝ => (u n + u r / x) / (n + r / x)) atTop (𝓝 ((u n + 0) / (n + 0))) :=
    (tendsto_const_nhds.add <| tendsto_const_nhds.div_atTop tendsto_id).div
      (tendsto_const_nhds.add <| tendsto_const_nhds.div_atTop tendsto_id) <| by simpa
  have B : Tendsto (fun x => (x * u n + u r) / (x * n + r)) atTop (𝓝 (u n / n)) := by
    rw [add_zero, add_zero] at A
    refine A.congr' <| (eventually_ne_atTop 0).mono fun x hx => ?_
    simp only [(· ∘ ·), add_div' _ _ _ hx, div_div_div_cancel_right₀ hx, mul_comm]
  /-
    u : Nat → Real
    h : Subadditive u
    L : Real
    n : Nat
    hn : Ne n 0
    hL : LT.lt (HDiv.hDiv (u n) ↑n) L
    r : Nat
    x✝ : LT.lt r n
    A : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (u n) (HDiv.hDiv (u r) x)) ( …
    B : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) (u r)) ( …
    ⊢ Filter.Eventually (fun a => LT.lt (HDiv.hDiv (u (HAdd.hAdd (HMul.hMul n a) r …
  -/
  refine ((B.comp tendsto_natCast_atTop_atTop).eventually (gt_mem_nhds hL)).mono fun k hk => ?_
  /- Finally, we use an upper estimate on `u (k * n + r)` to get an estimate on
  `u (k * n + r) / (k * n + r)`. -/
  /-
    u : Nat → Real
    h : Subadditive u
    L : Real
    n : Nat
    hn : Ne n 0
    hL : LT.lt (HDiv.hDiv (u n) ↑n) L
    r : Nat
    x✝ : LT.lt r n
    A : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (u n) (HDiv.hDiv (u r) x)) ( …
    B : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) (u r)) ( …
    k : Nat
    hk : LT.lt (Function.comp (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) ( …
    ⊢ LT.lt (HDiv.hDiv (u (HAdd.hAdd (HMul.hMul n k) r)) ↑(HAdd.hAdd (HMul.hMul n  …
  -/
  rw [mul_comm]
  /-
    u : Nat → Real
    h : Subadditive u
    L : Real
    n : Nat
    hn : Ne n 0
    hL : LT.lt (HDiv.hDiv (u n) ↑n) L
    r : Nat
    x✝ : LT.lt r n
    A : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (u n) (HDiv.hDiv (u r) x)) ( …
    B : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) (u r)) ( …
    k : Nat
    hk : LT.lt (Function.comp (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) ( …
    ⊢ LT.lt (HDiv.hDiv (u (HAdd.hAdd (HMul.hMul k n) r)) ↑(HAdd.hAdd (HMul.hMul k  …
  -/
  refine lt_of_le_of_lt ?_ hk
  /-
    u : Nat → Real
    h : Subadditive u
    L : Real
    n : Nat
    hn : Ne n 0
    hL : LT.lt (HDiv.hDiv (u n) ↑n) L
    r : Nat
    x✝ : LT.lt r n
    A : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (u n) (HDiv.hDiv (u r) x)) ( …
    B : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) (u r)) ( …
    k : Nat
    hk : LT.lt (Function.comp (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) ( …
    ⊢ LE.le (HDiv.hDiv (u (HAdd.hAdd (HMul.hMul k n) r)) ↑(HAdd.hAdd (HMul.hMul k  …
  -/
  simp only [(· ∘ ·), ← Nat.cast_add, ← Nat.cast_mul]
  /-
    u : Nat → Real
    h : Subadditive u
    L : Real
    n : Nat
    hn : Ne n 0
    hL : LT.lt (HDiv.hDiv (u n) ↑n) L
    r : Nat
    x✝ : LT.lt r n
    A : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (u n) (HDiv.hDiv (u r) x)) ( …
    B : Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) (u r)) ( …
    k : Nat
    hk : LT.lt (Function.comp (fun x => HDiv.hDiv (HAdd.hAdd (HMul.hMul x (u n)) ( …
    ⊢ LE.le (HDiv.hDiv (u (HAdd.hAdd (HMul.hMul k n) r)) ↑(HAdd.hAdd (HMul.hMul k  …
  -/
  exact div_le_div_of_nonneg_right (h.apply_mul_add_le _ _ _) (Nat.cast_nonneg _)
  /-
    🎉 no goals
  -/


/-- Fekete's lemma: a subadditive sequence which is bounded below converges. -/
theorem tendsto_lim (hbdd : BddBelow (range fun n => u n / n)) :
    Tendsto (fun n => u n / n) atTop (𝓝 h.lim) := by
  /-
    u : Nat → Real
    h : Subadditive u
    hbdd : BddBelow (Set.range fun n => HDiv.hDiv (u n) ↑n)
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (u n) ↑n) Filter.atTop (nhds h.lim)
  -/
  refine tendsto_order.2 ⟨fun l hl => ?_, fun L hL => ?_⟩
  · refine eventually_atTop.2
      ⟨1, fun n hn => hl.trans_le (h.lim_le_div hbdd (zero_lt_one.trans_le hn).ne')⟩
  · obtain ⟨n, npos, hn⟩ : ∃ n : ℕ, 0 < n ∧ u n / n < L := by
      rw [Subadditive.lim] at hL
      rcases exists_lt_of_csInf_lt (by simp) hL with ⟨x, hx, xL⟩
      rcases (mem_image _ _ _).1 hx with ⟨n, hn, rfl⟩
      exact ⟨n, zero_lt_one.trans_le hn, xL⟩
    /-
      case refine_2.intro.intro
      u : Nat → Real
      h : Subadditive u
      hbdd : BddBelow (Set.range fun n => HDiv.hDiv (u n) ↑n)
      L : Real
      hL : GT.gt L h.lim
      n : Nat
      npos : LT.lt 0 n
      hn : LT.lt (HDiv.hDiv (u n) ↑n) L
      ⊢ Filter.Eventually (fun b => LT.lt (HDiv.hDiv (u b) ↑b) L) Filter.atTop
    -/
    exact h.eventually_div_lt_of_div_lt npos.ne' hn
    /-
      🎉 no goals
    -/


