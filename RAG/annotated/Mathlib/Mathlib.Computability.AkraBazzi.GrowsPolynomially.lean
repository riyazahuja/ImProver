/-- The growth condition that the function `g` must satisfy for the Akra-Bazzi theorem to apply.
It roughly states that `c₁ g(n) ≤ g(u) ≤ c₂ g(n)`, for `u` between `b*n` and `n` for any
constant `b ∈ (0,1)`. -/
def GrowsPolynomially (f : ℝ → ℝ) : Prop :=
  ∀ b ∈ Set.Ioo 0 1, ∃ c₁ > 0, ∃ c₂ > 0,
    ∀ᶠ x in atTop, ∀ u ∈ Set.Icc (b * x) x, f u ∈ Set.Icc (c₁ * (f x)) (c₂ * f x)


lemma congr_of_eventuallyEq {f g : ℝ → ℝ} (hfg : f =ᶠ[atTop] g) (hg : GrowsPolynomially g) :
    GrowsPolynomially f := by
  /-
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    ⊢ AkraBazziRecurrence.GrowsPolynomially f
  -/
  intro b hb
  /-
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hg' := hg b hb
  /-
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hg' : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fi …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₁, hc₁_mem, c₂, hc₂_mem, hg'⟩ := hg'
  /-
    case intro.intro.intro.intro
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hg' : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨c₁, hc₁_mem, c₂, hc₂_mem, ?_⟩
  filter_upwards [hg', (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hfg, hfg]
    with x hx₁ hx₂ hx₃
  /-
    case h
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hg' : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hx₁ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership. …
    hx₂ : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → Eq (f y) (g y)
    hx₃ : Eq (f x) (g x)
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.mem  …
  -/
  intro u hu
  /-
    case h
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hg' : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hx₁ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership. …
    hx₂ : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → Eq (f y) (g y)
    hx₃ : Eq (f x) (g x)
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul c₁ (f x)) (HMul.hMul c₂ (f x))) (f u)
  -/
  rw [hx₂ u hu.1, hx₃]
  /-
    case h
    f g : Real → Real
    hfg : Filter.atTop.EventuallyEq f g
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hg' : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hx₁ : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership. …
    hx₂ : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → Eq (f y) (g y)
    hx₃ : Eq (f x) (g x)
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul c₁ (g x)) (HMul.hMul c₂ (g x))) (g u)
  -/
  exact hx₁ u hu
  /-
    🎉 no goals
  -/


lemma iff_eventuallyEq {f g : ℝ → ℝ} (h : f =ᶠ[atTop] g) :
    GrowsPolynomially f ↔ GrowsPolynomially g :=
  ⟨fun hf => congr_of_eventuallyEq h.symm hf, fun hg => congr_of_eventuallyEq h hg⟩


lemma eventually_atTop_le {b : ℝ} (hb : b ∈ Set.Ioo 0 1) (hf : GrowsPolynomially f) :
    ∃ c > 0, ∀ᶠ x in atTop, ∀ u ∈ Set.Icc (b * x) x, f u ≤ c * f x := by
  /-
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => ∀ (u : Real), M …
  -/
  obtain ⟨c₁, _, c₂, hc₂, h⟩ := hf b hb
  /-
    case intro.intro.intro.intro
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c₁ : Real
    left✝ : GT.gt c₁ 0
    c₂ : Real
    hc₂ : GT.gt c₂ 0
    h : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hM …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => ∀ (u : Real), M …
  -/
  refine ⟨c₂, hc₂, ?_⟩
  /-
    case intro.intro.intro.intro
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c₁ : Real
    left✝ : GT.gt c₁ 0
    c₂ : Real
    hc₂ : GT.gt c₂ 0
    h : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hM …
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  filter_upwards [h]
  /-
    case h
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c₁ : Real
    left✝ : GT.gt c₁ 0
    c₂ : Real
    hc₂ : GT.gt c₂ 0
    h : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hM …
    ⊢ ∀ (a : Real), (∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b a) a) u →  …
  -/
  exact fun _ H u hu => (H u hu).2
  /-
    🎉 no goals
  -/


lemma eventually_atTop_le_nat {b : ℝ} (hb : b ∈ Set.Ioo 0 1) (hf : GrowsPolynomially f) :
    ∃ c > 0, ∀ᶠ (n : ℕ) in atTop, ∀ u ∈ Set.Icc (b * n) n, f u ≤ c * f n := by
  /-
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (u : Real), M …
  -/
  obtain ⟨c, hc_mem, hc⟩ := hf.eventually_atTop_le hb
  /-
    case intro.intro
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c : Real
    hc_mem : GT.gt c 0
    hc : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (u : Real), M …
  -/
  exact ⟨c, hc_mem, hc.natCast_atTop⟩
  /-
    🎉 no goals
  -/


lemma eventually_atTop_ge {b : ℝ} (hb : b ∈ Set.Ioo 0 1) (hf : GrowsPolynomially f) :
    ∃ c > 0, ∀ᶠ x in atTop, ∀ u ∈ Set.Icc (b * x) x, c * f x ≤ f u := by
  /-
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => ∀ (u : Real), M …
  -/
  obtain ⟨c₁, hc₁, c₂, _, h⟩ := hf b hb
  /-
    case intro.intro.intro.intro
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    c₂ : Real
    left✝ : GT.gt c₂ 0
    h : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hM …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => ∀ (u : Real), M …
  -/
  refine ⟨c₁, hc₁, ?_⟩
  /-
    case intro.intro.intro.intro
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    c₂ : Real
    left✝ : GT.gt c₂ 0
    h : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hM …
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  filter_upwards [h]
  /-
    case h
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c₁ : Real
    hc₁ : GT.gt c₁ 0
    c₂ : Real
    left✝ : GT.gt c₂ 0
    h : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hM …
    ⊢ ∀ (a : Real), (∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b a) a) u →  …
  -/
  exact fun _ H u hu => (H u hu).1
  /-
    🎉 no goals
  -/


lemma eventually_atTop_ge_nat {b : ℝ} (hb : b ∈ Set.Ioo 0 1) (hf : GrowsPolynomially f) :
    ∃ c > 0, ∀ᶠ (n : ℕ) in atTop, ∀ u ∈ Set.Icc (b * n) n, c * f n ≤ f u := by
  /-
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (u : Real), M …
  -/
  obtain ⟨c, hc_mem, hc⟩ := hf.eventually_atTop_ge hb
  /-
    case intro.intro
    f : Real → Real
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : AkraBazziRecurrence.GrowsPolynomially f
    c : Real
    hc_mem : GT.gt c 0
    hc : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun n => ∀ (u : Real), M …
  -/
  exact ⟨c, hc_mem, hc.natCast_atTop⟩
  /-
    🎉 no goals
  -/


lemma eventually_zero_of_frequently_zero (hf : GrowsPolynomially f) (hf' : ∃ᶠ x in atTop, f x = 0) :
    ∀ᶠ x in atTop, f x = 0 := by
  /-
    f : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf' : Filter.Frequently (fun x => Eq (f x) 0) Filter.atTop
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) Filter.atTop
  -/
  obtain ⟨c₁, hc₁_mem, c₂, hc₂_mem, hf⟩ := hf (1/2) (by norm_num)
  /-
    case intro.intro.intro.intro
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hf' : Filter.Frequently (fun x => Eq (f x) 0) Filter.atTop
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) Filter.atTop
  -/
  rw [frequently_atTop] at hf'
  /-
    case intro.intro.intro.intro
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hf' : ∀ (a : Real), Exists fun b => And (GE.ge b a) (Eq (f b) 0)
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) Filter.atTop
  -/
  filter_upwards [eventually_forall_ge_atTop.mpr hf, eventually_gt_atTop 0] with x hx hx_pos
  /-
    case h
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hf' : ∀ (a : Real), Exists fun b => And (GE.ge b a) (Eq (f b) 0)
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    x : Real
    hx : ∀ (y : Real), LE.le x y → ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMu …
    hx_pos : LT.lt 0 x
    ⊢ Eq (f x) 0
  -/
  obtain ⟨x₀, hx₀_ge, hx₀⟩ := hf' (max x 1)
  have x₀_pos := calc
    0 < 1 := by norm_num
    _ ≤ x₀ := le_of_max_le_right hx₀_ge
  have hmain : ∀ (m : ℕ) (z : ℝ), x ≤ z →
      z ∈ Set.Icc ((2 : ℝ)^(-(m : ℤ) -1) * x₀) ((2 : ℝ)^(-(m : ℤ)) * x₀) → f z = 0 := by
    intro m
    induction m with
    | zero =>
      simp only [CharP.cast_eq_zero, neg_zero, zero_sub, zpow_zero, one_mul] at *
      specialize hx x₀ (le_of_max_le_left hx₀_ge)
      simp only [hx₀, mul_zero, Set.Icc_self, Set.mem_singleton_iff] at hx
      refine fun z _ hz => hx _ ?_
      simp only [zpow_neg, zpow_one] at hz
      simp only [one_div, hz]
    | succ k ih =>
      intro z hxz hz
      simp only [Nat.succ_eq_add_one, Nat.cast_add, Nat.cast_one] at *
      have hx' : x ≤ (2 : ℝ)^(-(k : ℤ) - 1) * x₀ := by
        calc x ≤ z := hxz
          _ ≤ _ := by simp only [neg_add, ← sub_eq_add_neg] at hz; exact hz.2
      specialize hx ((2 : ℝ)^(-(k : ℤ) - 1) * x₀) hx' z
      specialize ih ((2 : ℝ)^(-(k : ℤ) - 1) * x₀) hx' ?ineq
      case ineq =>
        rw [Set.left_mem_Icc]
        gcongr
        · norm_num
        · omega
      simp only [ih, mul_zero, Set.Icc_self, Set.mem_singleton_iff] at hx
      refine hx ⟨?lb₁, ?ub₁⟩
      case lb₁ =>
        rw [one_div, ← zpow_neg_one, ← mul_assoc, ← zpow_add₀ (by norm_num)]
        have h₁ : (-1 : ℤ)  + (-k - 1) = -k - 2 := by ring
        have h₂ : -(k + (1 : ℤ)) - 1 = -k - 2 := by ring
        rw [h₁]
        rw [h₂] at hz
        exact hz.1
      case ub₁ =>
        have := hz.2
        simp only [neg_add, ← sub_eq_add_neg] at this
        exact this
  /-
    case h.intro.intro
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hf' : ∀ (a : Real), Exists fun b => And (GE.ge b a) (Eq (f b) 0)
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    x : Real
    hx : ∀ (y : Real), LE.le x y → ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMu …
    hx_pos : LT.lt 0 x
    x₀ : Real
    hx₀_ge : GE.ge x₀ (Max.max x 1)
    hx₀ : Eq (f x₀) 0
    x₀_pos : LT.lt 0 x₀
    hmain : ∀ (m : Nat) (z : Real), LE.le x z → Membership.mem (Set.Icc (HMul.hMul …
    ⊢ Eq (f x) 0
  -/
  refine hmain ⌊-logb 2 (x / x₀)⌋₊ x le_rfl ⟨?lb, ?ub⟩
  case lb =>
    rw [← le_div_iff₀ x₀_pos]
    refine (logb_le_logb (b := 2) (by norm_num) (zpow_pos (by norm_num) _)
      (by positivity)).mp ?_
    rw [← rpow_intCast, logb_rpow (by norm_num) (by norm_num), ← neg_le_neg_iff]
    simp only [Int.cast_sub, Int.cast_neg, Int.cast_natCast, Int.cast_one, neg_sub, sub_neg_eq_add]
    calc -logb 2 (x/x₀) ≤ ⌈-logb 2 (x/x₀)⌉₊ := Nat.le_ceil (-logb 2 (x / x₀))
         _ ≤ _ := by rw [add_comm]; exact_mod_cast Nat.ceil_le_floor_add_one _
  case ub =>
    rw [← div_le_iff₀ x₀_pos]
    refine (logb_le_logb (b := 2) (by norm_num) (by positivity)
      (zpow_pos (by norm_num) _)).mp ?_
    rw [← rpow_intCast, logb_rpow (by norm_num) (by norm_num), ← neg_le_neg_iff]
    simp only [Int.cast_neg, Int.cast_natCast, neg_neg]
    have : 0 ≤ -logb 2 (x / x₀) := by
      rw [neg_nonneg]
      refine logb_nonpos (by norm_num) (by positivity) ?_
      rw [div_le_one x₀_pos]
      exact le_of_max_le_left hx₀_ge
    exact_mod_cast Nat.floor_le this


lemma eventually_atTop_nonneg_or_nonpos (hf : GrowsPolynomially f) :
    (∀ᶠ x in atTop, 0 ≤ f x) ∨ (∀ᶠ x in atTop, f x ≤ 0) := by
  /-
    f : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ Or (Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop) (Filter.Eventua …
  -/
  obtain ⟨c₁, _, c₂, _, h⟩ := hf (1/2) (by norm_num)
  match lt_trichotomy c₁ c₂ with
  | .inl hlt => -- c₁ < c₂
    left
    filter_upwards [h, eventually_ge_atTop 0] with x hx hx_nonneg
    have h' : 3 / 4 * x ∈ Set.Icc (1 / 2 * x) x := by
      rw [Set.mem_Icc]
      exact ⟨by gcongr ?_ * x; norm_num, by linarith⟩
    have hu := hx (3/4 * x) h'
    have hu := Set.nonempty_of_mem hu
    rw [Set.nonempty_Icc] at hu
    have hu' : 0 ≤ (c₂ - c₁) * f x := by linarith
    exact nonneg_of_mul_nonneg_right hu' (by linarith)
  | .inr (.inr hgt) => -- c₂ < c₁
    right
    filter_upwards [h, eventually_ge_atTop 0] with x hx hx_nonneg
    have h' : 3 / 4 * x ∈ Set.Icc (1 / 2 * x) x := by
      rw [Set.mem_Icc]
      exact ⟨by gcongr ?_ * x; norm_num, by linarith⟩
    have hu := hx (3/4 * x) h'
    have hu := Set.nonempty_of_mem hu
    rw [Set.nonempty_Icc] at hu
    have hu' : (c₁ - c₂) * f x ≤ 0 := by linarith
    exact nonpos_of_mul_nonpos_right hu' (by linarith)
  | .inr (.inl heq) => -- c₁ = c₂
    have hmain : ∃ c, ∀ᶠ x in atTop, f x = c := by
      simp only [heq, Set.Icc_self, Set.mem_singleton_iff, one_mul] at h
      rw [eventually_atTop] at h
      obtain ⟨n₀, hn₀⟩ := h
      refine ⟨f (max n₀ 2), ?_⟩
      rw [eventually_atTop]
      refine ⟨max n₀ 2, ?_⟩
      refine Real.induction_Ico_mul _ 2 (by norm_num) (by positivity) ?base ?step
      case base =>
        intro x ⟨hxlb, hxub⟩
        have h₁ := calc n₀ ≤ 1 * max n₀ 2 := by simp
                        _ ≤ 2 * max n₀ 2 := by gcongr; norm_num
        have h₂ := hn₀ (2 * max n₀ 2) h₁ (max n₀ 2) ⟨by simp [hxlb], by linarith⟩
        rw [h₂]
        exact hn₀ (2 * max n₀ 2) h₁ x ⟨by simp [hxlb], le_of_lt hxub⟩
      case step =>
        intro n hn hyp_ind z hz
        have z_nonneg : 0 ≤ z := by
          calc (0 : ℝ) ≤ (2 : ℝ)^n * max n₀ 2 := by
                        exact mul_nonneg (pow_nonneg (by norm_num) _) (by norm_num)
                  _ ≤ z := by exact_mod_cast hz.1
        have le_2n : max n₀ 2 ≤ (2 : ℝ)^n * max n₀ 2 := by
          nth_rewrite 1 [← one_mul (max n₀ 2)]
          gcongr
          exact one_le_pow₀ (by norm_num : (1 : ℝ) ≤ 2)
        have n₀_le_z : n₀ ≤ z := by
          calc n₀ ≤ max n₀ 2 := by simp
                _ ≤ (2 : ℝ)^n * max n₀ 2 := le_2n
                _ ≤ _ := by exact_mod_cast hz.1
        have fz_eq_c₂fz : f z = c₂ * f z := hn₀ z n₀_le_z z ⟨by linarith, le_rfl⟩
        have z_to_half_z' : f (1/2 * z) = c₂ * f z := hn₀ z n₀_le_z (1/2 * z) ⟨le_rfl, by linarith⟩
        have z_to_half_z : f (1/2 * z) = f z := by rwa [← fz_eq_c₂fz] at z_to_half_z'
        have half_z_to_base : f (1/2 * z) = f (max n₀ 2) := by
          refine hyp_ind (1/2 * z) ⟨?lb, ?ub⟩
          case lb =>
            calc max n₀ 2 ≤ ((1 : ℝ)/(2 : ℝ)) * (2 : ℝ) ^ 1 * max n₀ 2 := by simp
                        _ ≤ ((1 : ℝ)/(2 : ℝ)) * (2 : ℝ) ^ n * max n₀ 2 := by gcongr; norm_num
                        _ ≤ _ := by rw [mul_assoc]; gcongr; exact_mod_cast hz.1
          case ub =>
            have h₁ : (2 : ℝ)^n = ((1 : ℝ)/(2 : ℝ)) * (2 : ℝ)^(n+1) := by
              rw [one_div, pow_add, pow_one]
              ring
            rw [h₁, mul_assoc]
            gcongr
            exact_mod_cast hz.2
        rw [← z_to_half_z, half_z_to_base]
    obtain ⟨c, hc⟩ := hmain
    cases le_or_lt 0 c with
    | inl hpos =>
      exact Or.inl <| by filter_upwards [hc] with _ hc; simpa only [hc]
    | inr hneg =>
      right
      filter_upwards [hc] with x hc
      exact le_of_lt <| by simpa only [hc]


lemma eventually_atTop_zero_or_pos_or_neg (hf : GrowsPolynomially f) :
    (∀ᶠ x in atTop, f x = 0) ∨ (∀ᶠ x in atTop, 0 < f x) ∨ (∀ᶠ x in atTop, f x < 0) := by
  if h : ∃ᶠ x in atTop, f x = 0 then
    exact Or.inl <| eventually_zero_of_frequently_zero hf h
  else
    rw [not_frequently] at h
    push_neg at h
    cases eventually_atTop_nonneg_or_nonpos hf with
    | inl h' =>
      refine Or.inr (Or.inl ?_)
      simp only [lt_iff_le_and_ne]
      rw [eventually_and]
      exact ⟨h', by filter_upwards [h] with x hx; exact hx.symm⟩
    | inr h' =>
      refine Or.inr (Or.inr ?_)
      simp only [lt_iff_le_and_ne]
      rw [eventually_and]
      exact ⟨h', h⟩


protected lemma neg {f : ℝ → ℝ} (hf : GrowsPolynomially f) : GrowsPolynomially (-f) := by
  /-
    f : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ AkraBazziRecurrence.GrowsPolynomially (Neg.neg f)
  -/
  intro b hb
  /-
    f : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₁, hc₁_mem, c₂, hc₂_mem, hf⟩ := hf b hb
  /-
    case intro.intro.intro.intro
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨c₂, hc₂_mem, c₁, hc₁_mem, ?_⟩
  /-
    case intro.intro.intro.intro
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  filter_upwards [hf] with x hx
  /-
    case h
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    x : Real
    hx : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.mem  …
  -/
  intro u hu
  /-
    case h
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    x : Real
    hx : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul c₂ (Neg.neg f x)) (HMul.hMul c₁ (Neg.neg  …
  -/
  simp only [Pi.neg_apply, Set.neg_mem_Icc_iff, neg_mul_eq_mul_neg, neg_neg]
  /-
    case h
    f : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    x : Real
    hx : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul c₁ (f x)) (HMul.hMul c₂ (f x))) (f u)
  -/
  exact hx u hu
  /-
    🎉 no goals
  -/


protected lemma neg_iff {f : ℝ → ℝ} : GrowsPolynomially f ↔ GrowsPolynomially (-f) :=
                                  /-
                                    f : Real → Real
                                    hf : AkraBazziRecurrence.GrowsPolynomially (Neg.neg f)
                                    ⊢ AkraBazziRecurrence.GrowsPolynomially f
                                  -/
  ⟨fun hf => hf.neg, fun hf => by rw [← neg_neg f]; exact hf.neg⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


protected lemma abs (hf : GrowsPolynomially f) : GrowsPolynomially (fun x => |f x|) := by
  cases eventually_atTop_nonneg_or_nonpos hf with
  | inl hf' =>
    have hmain : f =ᶠ[atTop] fun x => |f x| := by
      filter_upwards [hf'] with x hx
      rw [abs_of_nonneg hx]
    rw [← iff_eventuallyEq hmain]
    exact hf
  | inr hf' =>
    have hmain : -f =ᶠ[atTop] fun x => |f x| := by
      filter_upwards [hf'] with x hx
      simp only [Pi.neg_apply, abs_of_nonpos hx]

    rw [← iff_eventuallyEq hmain]
    exact hf.neg


protected lemma norm (hf : GrowsPolynomially f) : GrowsPolynomially (fun x => ‖f x‖) := by
  /-
    f : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => Norm.norm (f x)
  -/
  simp only [norm_eq_abs]
  /-
    f : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => abs (f x)
  -/
  exact hf.abs
  /-
    🎉 no goals
  -/


lemma growsPolynomially_const {c : ℝ} : GrowsPolynomially (fun _ => c) := by
  /-
    c : Real
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => c
  -/
  refine fun _ _ => ⟨1, by norm_num, 1, by norm_num, ?_⟩
  /-
    c x✝¹ : Real
    x✝ : Membership.mem (Set.Ioo 0 1) x✝¹
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  filter_upwards [] with x
  /-
    case h
    c x✝¹ : Real
    x✝ : Membership.mem (Set.Ioo 0 1) x✝¹
    x : Real
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul x✝¹ x) x) u → Membership.me …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma growsPolynomially_id : GrowsPolynomially (fun x => x) := by
  /-
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => x
  -/
  intro b hb
  /-
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨b, hb.1, ?_⟩
  /-
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₂ => And (GT.gt c₂ 0) (Filter.Eventually (fun x => ∀ (u : Real), …
  -/
  refine ⟨1, by norm_num, ?_⟩
  /-
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  filter_upwards with x u hu
  /-
    case h
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    x u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul b x) (HMul.hMul 1 x)) u
  -/
  simp only [one_mul, gt_iff_lt, not_le, Set.mem_Icc]
  /-
    case h
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    x u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ And (LE.le (HMul.hMul b x) u) (LE.le u x)
  -/
  exact ⟨hu.1, hu.2⟩
  /-
    🎉 no goals
  -/


protected lemma GrowsPolynomially.mul {f g : ℝ → ℝ} (hf : GrowsPolynomially f)
    (hg : GrowsPolynomially g) : GrowsPolynomially fun x => f x * g x := by
  suffices GrowsPolynomially fun x => |f x| * |g x| by
    cases eventually_atTop_nonneg_or_nonpos hf with
    | inl hf' =>
      cases eventually_atTop_nonneg_or_nonpos hg with
      | inl hg' =>
        have hmain : (fun x => f x * g x) =ᶠ[atTop] fun x => |f x| * |g x| := by
          filter_upwards [hf', hg'] with x hx₁ hx₂
          rw [abs_of_nonneg hx₁, abs_of_nonneg hx₂]
        rwa [iff_eventuallyEq hmain]
      | inr hg' =>
        have hmain : (fun x => f x * g x) =ᶠ[atTop] fun x => -|f x| * |g x| := by
          filter_upwards [hf', hg'] with x hx₁ hx₂
          simp [abs_of_nonneg hx₁, abs_of_nonpos hx₂]
        simp only [iff_eventuallyEq hmain, neg_mul]
        exact this.neg
    | inr hf' =>
      cases eventually_atTop_nonneg_or_nonpos hg with
      | inl hg' =>
        have hmain : (fun x => f x * g x) =ᶠ[atTop] fun x => -|f x| * |g x| := by
          filter_upwards [hf', hg'] with x hx₁ hx₂
          rw [abs_of_nonpos hx₁, abs_of_nonneg hx₂, neg_neg]
        simp only [iff_eventuallyEq hmain, neg_mul]
        exact this.neg
      | inr hg' =>
        have hmain : (fun x => f x * g x) =ᶠ[atTop] fun x => |f x| * |g x| := by
          filter_upwards [hf', hg'] with x hx₁ hx₂
          simp [abs_of_nonpos hx₁, abs_of_nonpos hx₂]
        simp only [iff_eventuallyEq hmain, neg_mul]
        exact this
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HMul.hMul (abs (f x)) (abs (g …
  -/
  intro b hb
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hf := hf.abs b hb
  /-
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hg := hg.abs b hb
  /-
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    hg : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₁, hc₁_mem, c₂, hc₂_mem, hf⟩ := hf
  /-
    case intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hg : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₃, hc₃_mem, c₄, hc₄_mem, hg⟩ := hg
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    hc₄_mem : GT.gt c₄ 0
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨c₁ * c₃, by show 0 < c₁ * c₃; positivity, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    hc₄_mem : GT.gt c₄ 0
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₂ => And (GT.gt c₂ 0) (Filter.Eventually (fun x => ∀ (u : Real), …
  -/
  refine ⟨c₂ * c₄, by show 0 < c₂ * c₄; positivity, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    hc₄_mem : GT.gt c₄ 0
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  filter_upwards [hf, hg] with x hf hg
  /-
    case h
    f g : Real → Real
    hf✝¹ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝¹ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    hc₄_mem : GT.gt c₄ 0
    hg✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hf : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hg : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.mem  …
  -/
  intro u hu
  /-
    case h
    f g : Real → Real
    hf✝¹ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝¹ : AkraBazziRecurrence.GrowsPolynomially g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    hc₄_mem : GT.gt c₄ 0
    hg✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hf : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hg : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HMul.hMul c₁ c₃) (HMul.hMul (abs (f x))  …
  -/
  refine ⟨?lb, ?ub⟩
  case lb => calc
    c₁ * c₃ * (|f x| * |g x|) = (c₁ * |f x|) * (c₃ * |g x|) := by ring
    _ ≤ |f u| * |g u| := by
           gcongr
           · exact (hf u hu).1
           · exact (hg u hu).1
  case ub => calc
    |f u| * |g u| ≤ (c₂ * |f x|) * (c₄ * |g x|) := by
           gcongr
           · exact (hf u hu).2
           · exact (hg u hu).2
    _ = c₂ * c₄ * (|f x| * |g x|) := by ring


lemma GrowsPolynomially.const_mul {f : ℝ → ℝ} {c : ℝ} (hf : GrowsPolynomially f) :
    GrowsPolynomially fun x => c * f x :=
  GrowsPolynomially.mul growsPolynomially_const hf


protected lemma GrowsPolynomially.add {f g : ℝ → ℝ} (hf : GrowsPolynomially f)
    (hg : GrowsPolynomially g) (hf' : 0 ≤ᶠ[atTop] f) (hg' : 0 ≤ᶠ[atTop] g) :
    GrowsPolynomially fun x => f x + g x := by
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HAdd.hAdd (f x) (g x)
  -/
  intro b hb
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hf := hf b hb
  /-
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hg := hg b hb
  /-
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hf : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    hg : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₁, hc₁_mem, c₂, hc₂_mem, hf⟩ := hf
  /-
    case intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hg : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₃, hc₃_mem, c₄, _, hg⟩ := hg
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    left✝ : GT.gt c₄ 0
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨min c₁ c₃, by show 0 < min c₁ c₃; positivity, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hf✝ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf' : Filter.atTop.EventuallyLE 0 f
    hg' : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    left✝ : GT.gt c₄ 0
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₂ => And (GT.gt c₂ 0) (Filter.Eventually (fun x => ∀ (u : Real), …
  -/
  refine ⟨max c₂ c₄, by show 0 < max c₂ c₄; positivity, ?_⟩
  filter_upwards [hf, hg,
                  (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hf',
                  (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hg',
                  eventually_ge_atTop 0] with x hf hg hf' hg' hx_pos
  /-
    case h
    f g : Real → Real
    hf✝¹ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝¹ : AkraBazziRecurrence.GrowsPolynomially g
    hf'✝ : Filter.atTop.EventuallyLE 0 f
    hg'✝ : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    left✝ : GT.gt c₄ 0
    hg✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hf : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hg : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hf' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (f y)
    hg' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (g y)
    hx_pos : LE.le 0 x
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.mem  …
  -/
  intro u hu
  have hbx : b * x ≤ x := calc
    b * x ≤ 1 * x := by gcongr; exact le_of_lt hb.2
        _ = x := by ring
  /-
    case h
    f g : Real → Real
    hf✝¹ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝¹ : AkraBazziRecurrence.GrowsPolynomially g
    hf'✝ : Filter.atTop.EventuallyLE 0 f
    hg'✝ : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    left✝ : GT.gt c₄ 0
    hg✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hf : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hg : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hf' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (f y)
    hg' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (g y)
    hx_pos : LE.le 0 x
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    hbx : LE.le (HMul.hMul b x) x
    ⊢ Membership.mem (Set.Icc (HMul.hMul (Min.min c₁ c₃) (HAdd.hAdd (f x) (g x)))  …
  -/
  have fx_nonneg : 0 ≤ f x := hf' x hbx
  /-
    case h
    f g : Real → Real
    hf✝¹ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝¹ : AkraBazziRecurrence.GrowsPolynomially g
    hf'✝ : Filter.atTop.EventuallyLE 0 f
    hg'✝ : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    left✝ : GT.gt c₄ 0
    hg✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hf : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hg : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hf' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (f y)
    hg' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (g y)
    hx_pos : LE.le 0 x
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    hbx : LE.le (HMul.hMul b x) x
    fx_nonneg : LE.le 0 (f x)
    ⊢ Membership.mem (Set.Icc (HMul.hMul (Min.min c₁ c₃) (HAdd.hAdd (f x) (g x)))  …
  -/
  have gx_nonneg : 0 ≤ g x := hg' x hbx
  /-
    case h
    f g : Real → Real
    hf✝¹ : AkraBazziRecurrence.GrowsPolynomially f
    hg✝¹ : AkraBazziRecurrence.GrowsPolynomially g
    hf'✝ : Filter.atTop.EventuallyLE 0 f
    hg'✝ : Filter.atTop.EventuallyLE 0 g
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : GT.gt c₁ 0
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hf✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    c₃ : Real
    hc₃_mem : GT.gt c₃ 0
    c₄ : Real
    left✝ : GT.gt c₄ 0
    hg✝ : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul. …
    x : Real
    hf : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hg : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.m …
    hf' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (f y)
    hg' : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (0 y) (g y)
    hx_pos : LE.le 0 x
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    hbx : LE.le (HMul.hMul b x) x
    fx_nonneg : LE.le 0 (f x)
    gx_nonneg : LE.le 0 (g x)
    ⊢ Membership.mem (Set.Icc (HMul.hMul (Min.min c₁ c₃) (HAdd.hAdd (f x) (g x)))  …
  -/
  refine ⟨?lb, ?ub⟩
  case lb => calc
    min c₁ c₃ * (f x + g x) = min c₁ c₃ * f x + min c₁ c₃ * g x := by simp only [mul_add]
      _ ≤ c₁ * f x + c₃ * g x := by
              gcongr
              · exact min_le_left _ _
              · exact min_le_right _ _
      _ ≤ f u + g u := by
              gcongr
              · exact (hf u hu).1
              · exact (hg u hu).1
  case ub => calc
    max c₂ c₄ * (f x + g x) = max c₂ c₄ * f x + max c₂ c₄ * g x := by simp only [mul_add]
      _ ≥ c₂ * f x + c₄ * g x := by gcongr
                                    · exact le_max_left _ _
                                    · exact le_max_right _ _
      _ ≥ f u + g u := by gcongr
                          · exact (hf u hu).2
                          · exact (hg u hu).2


lemma GrowsPolynomially.add_isLittleO {f g : ℝ → ℝ} (hf : GrowsPolynomially f)
    (hfg : g =o[atTop] f) : GrowsPolynomially fun x => f x + g x := by
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hfg : Asymptotics.IsLittleO Filter.atTop g f
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HAdd.hAdd (f x) (g x)
  -/
  intro b hb
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hfg : Asymptotics.IsLittleO Filter.atTop g f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hb_ub := hb.2
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hfg : Asymptotics.IsLittleO Filter.atTop g f
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_ub : LT.lt b 1
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  rw [isLittleO_iff] at hfg
  cases hf.eventually_atTop_nonneg_or_nonpos with
  | inl hf' => -- f is eventually non-negative
    have hf := hf b hb
    obtain ⟨c₁, hc₁_mem : 0 < c₁, c₂, hc₂_mem : 0 < c₂, hf⟩ := hf
    specialize hfg (c := 1/2) (by norm_num)
    refine ⟨c₁ / 3, by positivity, 3*c₂, by positivity, ?_⟩
    filter_upwards [hf,
                    (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hfg,
                    (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hf',
                    eventually_ge_atTop 0] with x hf₁ hfg' hf₂ hx_nonneg
    have hbx : b * x ≤ x := by nth_rewrite 2 [← one_mul x]; gcongr
    have hfg₂ : ‖g x‖ ≤ 1/2 * f x := by
      calc ‖g x‖ ≤ 1/2 * ‖f x‖ := hfg' x hbx
           _ = 1/2 * f x := by congr; exact norm_of_nonneg (hf₂ _ hbx)
    have hx_ub : f x + g x ≤ 3/2 * f x := by
      calc _ ≤ f x + ‖g x‖ := by gcongr; exact le_norm_self (g x)
           _ ≤ f x + 1/2 * f x := by gcongr
           _ = 3/2 * f x := by ring
    have hx_lb : 1/2 * f x ≤ f x + g x := by
      calc f x + g x ≥ f x - ‖g x‖ := by
                rw [sub_eq_add_neg, norm_eq_abs]; gcongr; exact neg_abs_le (g x)
           _ ≥ f x - 1/2 * f x := by gcongr
           _ = 1/2 * f x := by ring
    intro u ⟨hu_lb, hu_ub⟩
    have hfu_nonneg : 0 ≤ f u := hf₂ _ hu_lb
    have hfg₃ : ‖g u‖ ≤ 1/2 * f u := by
      calc ‖g u‖ ≤ 1/2 * ‖f u‖ := hfg' _ hu_lb
           _ = 1/2 * f u := by congr; simp only [norm_eq_abs, abs_eq_self, hfu_nonneg]
    refine ⟨?lb, ?ub⟩
    case lb =>
      calc f u + g u ≥ f u - ‖g u‖ := by
                  rw [sub_eq_add_neg, norm_eq_abs]; gcongr; exact neg_abs_le _
           _ ≥ f u - 1/2 * f u := by gcongr
           _ = 1/2 * f u := by ring
           _ ≥ 1/2 * (c₁ * f x) := by gcongr; exact (hf₁ u ⟨hu_lb, hu_ub⟩).1
           _ = c₁/3 * (3/2 * f x) := by ring
           _ ≥ c₁/3 * (f x + g x) := by gcongr
    case ub =>
      calc _ ≤ f u + ‖g u‖ := by gcongr; exact le_norm_self (g u)
           _ ≤ f u + 1/2 * f u := by gcongr
           _ = 3/2 * f u := by ring
           _ ≤ 3/2 * (c₂ * f x) := by gcongr; exact (hf₁ u ⟨hu_lb, hu_ub⟩).2
           _ = 3*c₂ * (1/2 * f x) := by ring
           _ ≤ 3*c₂ * (f x + g x) := by gcongr
  | inr hf' => -- f is eventually nonpos
    have hf := hf b hb
    obtain ⟨c₁, hc₁_mem : 0 < c₁, c₂, hc₂_mem : 0 < c₂, hf⟩ := hf
    specialize hfg (c := 1/2) (by norm_num)
    refine ⟨3*c₁, by positivity, c₂/3, by positivity, ?_⟩
    filter_upwards [hf,
                    (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hfg,
                    (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hf',
                    eventually_ge_atTop 0] with x hf₁ hfg' hf₂ hx_nonneg
    have hbx : b * x ≤ x := by nth_rewrite 2 [← one_mul x]; gcongr
    have hfg₂ : ‖g x‖ ≤ -1/2 * f x := by
      calc ‖g x‖ ≤ 1/2 * ‖f x‖ := hfg' x hbx
           _ = 1/2 * (-f x) := by congr; exact norm_of_nonpos (hf₂ x hbx)
           _ = _ := by ring
    have hx_ub : f x + g x ≤ 1/2 * f x := by
      calc _ ≤ f x + ‖g x‖ := by gcongr; exact le_norm_self (g x)
           _ ≤ f x + (-1/2 * f x) := by gcongr
           _ = 1/2 * f x := by ring
    have hx_lb : 3/2 * f x ≤ f x + g x := by
      calc f x + g x ≥ f x - ‖g x‖ := by
                rw [sub_eq_add_neg, norm_eq_abs]; gcongr; exact neg_abs_le (g x)
           _ ≥ f x + 1/2 * f x := by
                  rw [sub_eq_add_neg]
                  gcongr
                  refine le_of_neg_le_neg ?bc.a
                  rwa [neg_neg, ← neg_mul, ← neg_div]
           _ = 3/2 * f x := by ring
    intro u ⟨hu_lb, hu_ub⟩
    have hfu_nonpos : f u ≤ 0 := hf₂ _ hu_lb
    have hfg₃ : ‖g u‖ ≤ -1/2 * f u := by
      calc ‖g u‖ ≤ 1/2 * ‖f u‖ := hfg' _ hu_lb
           _ = 1/2 * (-f u) := by congr; exact norm_of_nonpos hfu_nonpos
           _ = -1/2 * f u := by ring
    refine ⟨?lb, ?ub⟩
    case lb =>
      calc f u + g u ≥ f u - ‖g u‖ := by
                  rw [sub_eq_add_neg, norm_eq_abs]; gcongr; exact neg_abs_le _
           _ ≥ f u + 1/2 * f u := by
                  rw [sub_eq_add_neg]
                  gcongr
                  refine le_of_neg_le_neg ?_
                  rwa [neg_neg, ← neg_mul, ← neg_div]
           _ = 3/2 * f u := by ring
           _ ≥ 3/2 * (c₁ * f x) := by gcongr; exact (hf₁ u ⟨hu_lb, hu_ub⟩).1
           _ = 3*c₁ * (1/2 * f x) := by ring
           _ ≥ 3*c₁ * (f x + g x) := by gcongr
    case ub =>
      calc _ ≤ f u + ‖g u‖ := by gcongr; exact le_norm_self (g u)
           _ ≤ f u - 1/2 * f u := by
                rw [sub_eq_add_neg]
                gcongr
                rwa [← neg_mul, ← neg_div]
           _ = 1/2 * f u := by ring
           _ ≤ 1/2 * (c₂ * f x) := by gcongr; exact (hf₁ u ⟨hu_lb, hu_ub⟩).2
           _ = c₂/3 * (3/2 * f x) := by ring
           _ ≤ c₂/3 * (f x + g x) := by gcongr


protected lemma GrowsPolynomially.inv {f : ℝ → ℝ} (hf : GrowsPolynomially f) :
    GrowsPolynomially fun x => (f x)⁻¹ := by
  cases hf.eventually_atTop_zero_or_pos_or_neg with
  | inl hf' =>
    refine fun b hb => ⟨1, by simp, 1, by simp, ?_⟩
    have hb_pos := hb.1
    filter_upwards [hf', (tendsto_id.const_mul_atTop hb_pos).eventually_forall_ge_atTop hf']
      with x hx hx'
    intro u hu
    simp only [hx, inv_zero, mul_zero, Set.Icc_self, Set.mem_singleton_iff, hx' u hu.1]
  | inr hf_pos_or_neg =>
    suffices GrowsPolynomially fun x => |(f x)⁻¹| by
      cases hf_pos_or_neg with
      | inl hf' =>
        have hmain : (fun x => (f x)⁻¹) =ᶠ[atTop] fun x => |(f x)⁻¹| := by
          filter_upwards [hf'] with x hx₁
          rw [abs_of_nonneg (inv_nonneg_of_nonneg (le_of_lt hx₁))]
        rwa [iff_eventuallyEq hmain]
      | inr hf' =>
        have hmain : (fun x => (f x)⁻¹) =ᶠ[atTop] fun x => -|(f x)⁻¹| := by
          filter_upwards [hf'] with x hx₁
          simp [abs_of_nonpos (inv_nonpos.mpr (le_of_lt hx₁))]
        rw [iff_eventuallyEq hmain]
        exact this.neg
    have hf' : ∀ᶠ x in atTop, f x ≠ 0 := by
      cases hf_pos_or_neg with
      | inl H => filter_upwards [H] with _ hx; exact (ne_of_lt hx).symm
      | inr H => filter_upwards [H] with _ hx; exact (ne_of_gt hx).symm
    simp only [abs_inv]
    have hf := hf.abs
    intro b hb
    have hb_pos := hb.1
    obtain ⟨c₁, hc₁_mem, c₂, hc₂_mem, hf⟩ := hf b hb
    refine ⟨c₂⁻¹, by show 0 < c₂⁻¹; positivity, ?_⟩
    refine ⟨c₁⁻¹, by show 0 < c₁⁻¹; positivity, ?_⟩
    filter_upwards [hf, hf', (tendsto_id.const_mul_atTop hb_pos).eventually_forall_ge_atTop hf']
      with x hx hx' hx''
    intro u hu
    have h₁ : 0 < |f u| := by rw [abs_pos]; exact hx'' u hu.1
    refine ⟨?lb, ?ub⟩
    case lb =>
      rw [← mul_inv]
      gcongr
      exact (hx u hu).2
    case ub =>
      rw [← mul_inv]
      gcongr
      exact (hx u hu).1


protected lemma GrowsPolynomially.div {f g : ℝ → ℝ} (hf : GrowsPolynomially f)
    (hg : GrowsPolynomially g) : GrowsPolynomially fun x => f x / g x := by
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HDiv.hDiv (f x) (g x)
  -/
  have : (fun x => f x / g x) = fun x => f x * (g x)⁻¹ := by ext; rw [div_eq_mul_inv]
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    this : Eq (fun x => HDiv.hDiv (f x) (g x)) fun x => HMul.hMul (f x) (Inv.inv ( …
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HDiv.hDiv (f x) (g x)
  -/
  rw [this]
  /-
    f g : Real → Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hg : AkraBazziRecurrence.GrowsPolynomially g
    this : Eq (fun x => HDiv.hDiv (f x) (g x)) fun x => HMul.hMul (f x) (Inv.inv ( …
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HMul.hMul (f x) (Inv.inv (g x))
  -/
  exact GrowsPolynomially.mul hf (GrowsPolynomially.inv hg)
  /-
    🎉 no goals
  -/


protected lemma GrowsPolynomially.rpow (p : ℝ) (hf : GrowsPolynomially f)
    (hf_nonneg : ∀ᶠ x in atTop, 0 ≤ f x) : GrowsPolynomially fun x => (f x) ^ p := by
  /-
    f : Real → Real
    p : Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HPow.hPow (f x) p
  -/
  intro b hb
  /-
    f : Real → Real
    p : Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₁, (hc₁_mem : 0 < c₁), c₂, hc₂_mem, hfnew⟩ := hf b hb
  /-
    case intro.intro.intro.intro
    f : Real → Real
    p : Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : LT.lt 0 c₁
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hfnew : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMu …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hc₁p : 0 < c₁ ^ p := Real.rpow_pos_of_pos hc₁_mem _
  /-
    case intro.intro.intro.intro
    f : Real → Real
    p : Real
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    c₁ : Real
    hc₁_mem : LT.lt 0 c₁
    c₂ : Real
    hc₂_mem : GT.gt c₂ 0
    hfnew : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMu …
    hc₁p : LT.lt 0 (HPow.hPow c₁ p)
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hc₂p : 0 < c₂ ^ p := Real.rpow_pos_of_pos hc₂_mem _
  cases le_or_lt 0 p with
  | inl => -- 0 ≤ p
    refine ⟨c₁^p, hc₁p, ?_⟩
    refine ⟨c₂^p, hc₂p, ?_⟩
    filter_upwards [eventually_gt_atTop 0, hfnew, hf_nonneg,
        (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hf_nonneg]
        with x _ hf₁ hf_nonneg hf_nonneg₂
    intro u hu
    have fu_nonneg : 0 ≤ f u := hf_nonneg₂ u hu.1
    refine ⟨?lb, ?ub⟩
    case lb => calc
      c₁^p * (f x)^p = (c₁ * f x)^p := by rw [mul_rpow (le_of_lt hc₁_mem) hf_nonneg]
        _ ≤ _ := by gcongr; exact (hf₁ u hu).1
    case ub => calc
      (f u)^p ≤ (c₂ * f x)^p := by gcongr; exact (hf₁ u hu).2
        _ = _ := by rw [← mul_rpow (le_of_lt hc₂_mem) hf_nonneg]
  | inr hp => -- p < 0
    match hf.eventually_atTop_zero_or_pos_or_neg with
    | .inl hzero => -- eventually zero
      refine ⟨1, by norm_num, 1, by norm_num, ?_⟩
      filter_upwards [hzero, hfnew] with x hx hx'
      intro u hu
      simp only [hx, ne_eq, zero_rpow (ne_of_lt hp), mul_zero, le_refl, not_true, lt_self_iff_false,
        Set.Icc_self, Set.mem_singleton_iff]
      simp only [hx, mul_zero, Set.Icc_self, Set.mem_singleton_iff] at hx'
      rw [hx' u hu, zero_rpow (ne_of_lt hp)]
    | .inr (.inl hpos) => -- eventually positive
      refine ⟨c₂^p, hc₂p, ?_⟩
      refine ⟨c₁^p, hc₁p, ?_⟩
      filter_upwards [eventually_gt_atTop 0, hfnew, hpos,
          (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop hpos]
          with x _ hf₁ hf_pos hf_pos₂
      intro u hu
      refine ⟨?lb, ?ub⟩
      case lb => calc
        c₂^p * (f x)^p = (c₂ * f x)^p := by rw [mul_rpow (le_of_lt hc₂_mem) (le_of_lt hf_pos)]
          _ ≤ _ := rpow_le_rpow_of_exponent_nonpos (hf_pos₂ u hu.1) (hf₁ u hu).2 (le_of_lt hp)
      case ub => calc
        (f u)^p ≤ (c₁ * f x)^p := by
              exact rpow_le_rpow_of_exponent_nonpos (by positivity) (hf₁ u hu).1 (le_of_lt hp)
          _ = _ := by rw [← mul_rpow (le_of_lt hc₁_mem) (le_of_lt hf_pos)]
    | .inr (.inr hneg) => -- eventually negative (which is impossible)
      have : ∀ᶠ (_ : ℝ) in atTop, False := by
        filter_upwards [hf_nonneg, hneg] with x hx hx'; linarith
      rw [Filter.eventually_false_iff_eq_bot] at this
      exact False.elim <| (atTop_neBot).ne this


protected lemma GrowsPolynomially.pow (p : ℕ) (hf : GrowsPolynomially f)
    (hf_nonneg : ∀ᶠ x in atTop, 0 ≤ f x) : GrowsPolynomially fun x => (f x) ^ p := by
  /-
    f : Real → Real
    p : Nat
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HPow.hPow (f x) p
  -/
  simp_rw [← rpow_natCast]
  /-
    f : Real → Real
    p : Nat
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HPow.hPow (f x) ↑p
  -/
  exact hf.rpow p hf_nonneg
  /-
    🎉 no goals
  -/


protected lemma GrowsPolynomially.zpow (p : ℤ) (hf : GrowsPolynomially f)
    (hf_nonneg : ∀ᶠ x in atTop, 0 ≤ f x) : GrowsPolynomially fun x => (f x) ^ p := by
  /-
    f : Real → Real
    p : Int
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HPow.hPow (f x) p
  -/
  simp_rw [← rpow_intCast]
  /-
    f : Real → Real
    p : Int
    hf : AkraBazziRecurrence.GrowsPolynomially f
    hf_nonneg : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    ⊢ AkraBazziRecurrence.GrowsPolynomially fun x => HPow.hPow (f x) ↑p
  -/
  exact hf.rpow p hf_nonneg
  /-
    🎉 no goals
  -/


lemma growsPolynomially_rpow (p : ℝ) : GrowsPolynomially fun x => x ^ p :=
  (growsPolynomially_id).rpow p (eventually_ge_atTop 0)


lemma growsPolynomially_pow (p : ℕ) : GrowsPolynomially fun x => x ^ p :=
  (growsPolynomially_id).pow p (eventually_ge_atTop 0)


lemma growsPolynomially_zpow (p : ℤ) : GrowsPolynomially fun x => x ^ p :=
  (growsPolynomially_id).zpow p (eventually_ge_atTop 0)


lemma growsPolynomially_log : GrowsPolynomially Real.log := by
  /-
    ⊢ AkraBazziRecurrence.GrowsPolynomially Real.log
  -/
  intro b hb
  /-
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hb₀ : 0 < b := hb.1
  /-
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb₀ : LT.lt 0 b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨1 / 2, by norm_num, ?_⟩
  /-
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb₀ : LT.lt 0 b
    ⊢ Exists fun c₂ => And (GT.gt c₂ 0) (Filter.Eventually (fun x => ∀ (u : Real), …
  -/
  refine ⟨1, by norm_num, ?_⟩
  have h_tendsto : Tendsto (fun x => 1 / 2 * Real.log x) atTop atTop :=
    Tendsto.const_mul_atTop (by norm_num) Real.tendsto_log_atTop
  filter_upwards [eventually_gt_atTop 1,
                  (tendsto_id.const_mul_atTop hb.1).eventually_forall_ge_atTop
                    <| h_tendsto.eventually (eventually_gt_atTop (-Real.log b)) ] with x hx_pos hx
  /-
    case h
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb₀ : LT.lt 0 b
    h_tendsto : Filter.Tendsto (fun x => HMul.hMul (1 / 2) (Real.log x)) Filter.at …
    x : Real
    hx_pos : LT.lt 1 x
    hx : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LT.lt (Neg.neg (Real.log b)) …
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.mem  …
  -/
  intro u hu
  /-
    case h
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb₀ : LT.lt 0 b
    h_tendsto : Filter.Tendsto (fun x => HMul.hMul (1 / 2) (Real.log x)) Filter.at …
    x : Real
    hx_pos : LT.lt 1 x
    hx : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LT.lt (Neg.neg (Real.log b)) …
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    ⊢ Membership.mem (Set.Icc (HMul.hMul (1 / 2) (Real.log x)) (HMul.hMul 1 (Real. …
  -/
  refine ⟨?lb, ?ub⟩
  case lb => calc
    1 / 2 * Real.log x = Real.log x + (-1 / 2) * Real.log x := by ring
      _ ≤ Real.log x + Real.log b := by
              gcongr
              rw [neg_div, neg_mul, ← neg_le]
              refine le_of_lt (hx x ?_)
              calc b * x ≤ 1 * x := by gcongr; exact le_of_lt hb.2
                       _ = x := by rw [one_mul]
      _ = Real.log (b * x) := by rw [← Real.log_mul (by positivity) (by positivity), mul_comm]
      _ ≤ Real.log u := by gcongr; exact hu.1
  case ub =>
    rw [one_mul]
    gcongr
    · calc 0 < b * x := by positivity
         _ ≤ u := by exact hu.1
    · exact hu.2


lemma GrowsPolynomially.of_isTheta {f g : ℝ → ℝ} (hg : GrowsPolynomially g) (hf : f =Θ[atTop] g)
    (hf' : ∀ᶠ x in atTop, 0 ≤ f x) : GrowsPolynomially f := by
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    ⊢ AkraBazziRecurrence.GrowsPolynomially f
  -/
  intro b hb
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hb_pos := hb.1
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hf_lb := isBigO_iff''.mp hf.isBigO_symm
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    hf_lb : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (HM …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hf_ub := isBigO_iff'.mp hf.isBigO
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    hf_lb : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (HM …
    hf_ub : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (No …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₁, hc₁_pos : 0 < c₁, hf_lb⟩ := hf_lb
  /-
    case intro.intro
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    hf_ub : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (No …
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₂, hc₂_pos : 0 < c₂, hf_ub⟩ := hf_ub
  /-
    case intro.intro.intro.intro
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have hg := hg.norm b hb
  /-
    case intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    hg : Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Fil …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₃, hc₃_pos : 0 < c₃, hg⟩ := hg
  /-
    case intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    hg : Exists fun c₂ => And (GT.gt c₂ 0) (Filter.Eventually (fun x => ∀ (u : Rea …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  obtain ⟨c₄, hc₄_pos : 0 < c₄, hg⟩ := hg
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have h_lb_pos : 0 < c₁ * c₂⁻¹ * c₃ := by positivity
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  have h_ub_pos : 0 < c₂ * c₄ * c₁⁻¹ := by positivity
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    ⊢ Exists fun c₁ => And (GT.gt c₁ 0) (Exists fun c₂ => And (GT.gt c₂ 0) (Filter …
  -/
  refine ⟨c₁ * c₂⁻¹ * c₃, h_lb_pos, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    ⊢ Exists fun c₂_1 => And (GT.gt c₂_1 0) (Filter.Eventually (fun x => ∀ (u : Re …
  -/
  refine ⟨c₂ * c₄ * c₁⁻¹, h_ub_pos, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  have c₂_cancel : c₂⁻¹ * c₂ = 1 := inv_mul_cancel₀ (by positivity)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    c₂_cancel : Eq (HMul.hMul (Inv.inv c₂) c₂) 1
    ⊢ Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul …
  -/
  have c₁_cancel : c₁⁻¹ * c₁ = 1 := inv_mul_cancel₀ (by positivity)
  filter_upwards [(tendsto_id.const_mul_atTop hb_pos).eventually_forall_ge_atTop hf',
                  (tendsto_id.const_mul_atTop hb_pos).eventually_forall_ge_atTop hf_lb,
                  (tendsto_id.const_mul_atTop hb_pos).eventually_forall_ge_atTop hf_ub,
                  (tendsto_id.const_mul_atTop hb_pos).eventually_forall_ge_atTop hg,
                  eventually_ge_atTop 0]
    with x hf_pos h_lb h_ub hg_bound hx_pos
  /-
    case h
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    c₂_cancel : Eq (HMul.hMul (Inv.inv c₂) c₂) 1
    c₁_cancel : Eq (HMul.hMul (Inv.inv c₁) c₁) 1
    x : Real
    hf_pos : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le 0 (f y)
    h_lb : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (HMul.hMul c₁ (Norm. …
    h_ub : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (Norm.norm (f y)) (H …
    hg_bound : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → ∀ (u : Real), Membersh …
    hx_pos : LE.le 0 x
    ⊢ ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Membership.mem  …
  -/
  intro u hu
  have hbx : b * x ≤ x :=
    calc b * x ≤ 1 * x    := by gcongr; exact le_of_lt hb.2
             _ = x        := by rw [one_mul]
  /-
    case h
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    c₂_cancel : Eq (HMul.hMul (Inv.inv c₂) c₂) 1
    c₁_cancel : Eq (HMul.hMul (Inv.inv c₁) c₁) 1
    x : Real
    hf_pos : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le 0 (f y)
    h_lb : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (HMul.hMul c₁ (Norm. …
    h_ub : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (Norm.norm (f y)) (H …
    hg_bound : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → ∀ (u : Real), Membersh …
    hx_pos : LE.le 0 x
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    hbx : LE.le (HMul.hMul b x) x
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃ …
  -/
  have hg_bound := hg_bound x hbx
  /-
    case h
    f g : Real → Real
    hg✝ : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsTheta Filter.atTop f g
    hf' : Filter.Eventually (fun x => LE.le 0 (f x)) Filter.atTop
    b : Real
    hb : Membership.mem (Set.Ioo 0 1) b
    hb_pos : LT.lt 0 b
    c₁ : Real
    hc₁_pos : LT.lt 0 c₁
    hf_lb : Filter.Eventually (fun x => LE.le (HMul.hMul c₁ (Norm.norm (g x))) (No …
    c₂ : Real
    hc₂_pos : LT.lt 0 c₂
    hf_ub : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c₂ (Nor …
    c₃ : Real
    hc₃_pos : LT.lt 0 c₃
    c₄ : Real
    hc₄_pos : LT.lt 0 c₄
    hg : Filter.Eventually (fun x => ∀ (u : Real), Membership.mem (Set.Icc (HMul.h …
    h_lb_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃)
    h_ub_pos : LT.lt 0 (HMul.hMul (HMul.hMul c₂ c₄) (Inv.inv c₁))
    c₂_cancel : Eq (HMul.hMul (Inv.inv c₂) c₂) 1
    c₁_cancel : Eq (HMul.hMul (Inv.inv c₁) c₁) 1
    x : Real
    hf_pos : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le 0 (f y)
    h_lb : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (HMul.hMul c₁ (Norm. …
    h_ub : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → LE.le (Norm.norm (f y)) (H …
    hg_bound✝ : ∀ (y : Real), LE.le (HMul.hMul b (id x)) y → ∀ (u : Real), Members …
    hx_pos : LE.le 0 x
    u : Real
    hu : Membership.mem (Set.Icc (HMul.hMul b x) x) u
    hbx : LE.le (HMul.hMul b x) x
    hg_bound : ∀ (u : Real), Membership.mem (Set.Icc (HMul.hMul b x) x) u → Member …
    ⊢ Membership.mem (Set.Icc (HMul.hMul (HMul.hMul (HMul.hMul c₁ (Inv.inv c₂)) c₃ …
  -/
  refine ⟨?lb, ?ub⟩
  case lb => calc
    c₁ * c₂⁻¹ * c₃ * f x ≤ c₁ * c₂⁻¹ * c₃ * (c₂ * ‖g x‖) := by
          rw [← Real.norm_of_nonneg (hf_pos x hbx)]; gcongr; exact h_ub x hbx
      _ = (c₂⁻¹ * c₂) * c₁ * (c₃ * ‖g x‖) := by ring
      _ = c₁ * (c₃ * ‖g x‖) := by simp [c₂_cancel]
      _ ≤ c₁ * ‖g u‖ := by gcongr; exact (hg_bound u hu).1
      _ ≤ f u := by
          rw [← Real.norm_of_nonneg (hf_pos u hu.1)]
          exact h_lb u hu.1
  case ub => calc
    f u ≤ c₂ * ‖g u‖ := by rw [← Real.norm_of_nonneg (hf_pos u hu.1)]; exact h_ub u hu.1
      _ ≤ c₂ * (c₄ * ‖g x‖) := by gcongr; exact (hg_bound u hu).2
      _ = c₂ * c₄ * (c₁⁻¹ * c₁) * ‖g x‖ := by simp [c₁_cancel]; ring
      _ = c₂ * c₄ * c₁⁻¹ * (c₁ * ‖g x‖) := by ring
      _ ≤ c₂ * c₄ * c₁⁻¹ * f x := by
                gcongr
                rw [← Real.norm_of_nonneg (hf_pos x hbx)]
                exact h_lb x hbx


lemma GrowsPolynomially.of_isEquivalent {f g : ℝ → ℝ} (hg : GrowsPolynomially g)
    (hf : f ~[atTop] g) : GrowsPolynomially f := by
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsEquivalent Filter.atTop f g
    ⊢ AkraBazziRecurrence.GrowsPolynomially f
  -/
  have : f = g + (f - g) := by ext; simp
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsEquivalent Filter.atTop f g
    this : Eq f (HAdd.hAdd g (HSub.hSub f g))
    ⊢ AkraBazziRecurrence.GrowsPolynomially f
  -/
  rw [this]
  /-
    f g : Real → Real
    hg : AkraBazziRecurrence.GrowsPolynomially g
    hf : Asymptotics.IsEquivalent Filter.atTop f g
    this : Eq f (HAdd.hAdd g (HSub.hSub f g))
    ⊢ AkraBazziRecurrence.GrowsPolynomially (HAdd.hAdd g (HSub.hSub f g))
  -/
  exact add_isLittleO hg hf
  /-
    🎉 no goals
  -/


lemma GrowsPolynomially.of_isEquivalent_const {f : ℝ → ℝ} {c : ℝ} (hf : f ~[atTop] fun _ => c) :
    GrowsPolynomially f :=
  of_isEquivalent growsPolynomially_const hf


