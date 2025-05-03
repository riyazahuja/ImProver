local notation "expR" => Real.exp


/-- An auxiliary lemma that combines two double exponential estimates into a similar estimate
on the difference of the functions. -/
theorem isBigO_sub_exp_exp {a : ℝ} {f g : ℂ → E} {l : Filter ℂ} {u : ℂ → ℝ}
    (hBf : ∃ c < a, ∃ B, f =O[l] fun z => expR (B * expR (c * |u z|)))
    (hBg : ∃ c < a, ∃ B, g =O[l] fun z => expR (B * expR (c * |u z|))) :
    ∃ c < a, ∃ B, (f - g) =O[l] fun z => expR (B * expR (c * |u z|)) := by
  have : ∀ {c₁ c₂ B₁ B₂}, c₁ ≤ c₂ → 0 ≤ B₂ → B₁ ≤ B₂ → ∀ z,
      ‖expR (B₁ * expR (c₁ * |u z|))‖ ≤ ‖expR (B₂ * expR (c₂ * |u z|))‖ := fun hc hB₀ hB z ↦ by
    simp only [Real.norm_eq_abs, Real.abs_exp]; gcongr
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : Real
    f g : Complex → E
    l : Filter Complex
    u : Complex → Real
    hBf : Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO l f  …
    hBg : Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO l g  …
    this : ∀ {c₁ c₂ B₁ B₂ : Real}, LE.le c₁ c₂ → LE.le 0 B₂ → LE.le B₁ B₂ → ∀ (z : …
    ⊢ Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO l (HSub. …
  -/
  rcases hBf with ⟨cf, hcf, Bf, hOf⟩; rcases hBg with ⟨cg, hcg, Bg, hOg⟩
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : Real
    f g : Complex → E
    l : Filter Complex
    u : Complex → Real
    this : ∀ {c₁ c₂ B₁ B₂ : Real}, LE.le c₁ c₂ → LE.le 0 B₂ → LE.le B₁ B₂ → ∀ (z : …
    cf : Real
    hcf : LT.lt cf a
    Bf : Real
    hOf : Asymptotics.IsBigO l f fun z => Real.exp (HMul.hMul Bf (Real.exp (HMul.h …
    cg : Real
    hcg : LT.lt cg a
    Bg : Real
    hOg : Asymptotics.IsBigO l g fun z => Real.exp (HMul.hMul Bg (Real.exp (HMul.h …
    ⊢ Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO l (HSub. …
  -/
  refine ⟨max cf cg, max_lt hcf hcg, max 0 (max Bf Bg), ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : Real
    f g : Complex → E
    l : Filter Complex
    u : Complex → Real
    this : ∀ {c₁ c₂ B₁ B₂ : Real}, LE.le c₁ c₂ → LE.le 0 B₂ → LE.le B₁ B₂ → ∀ (z : …
    cf : Real
    hcf : LT.lt cf a
    Bf : Real
    hOf : Asymptotics.IsBigO l f fun z => Real.exp (HMul.hMul Bf (Real.exp (HMul.h …
    cg : Real
    hcg : LT.lt cg a
    Bg : Real
    hOg : Asymptotics.IsBigO l g fun z => Real.exp (HMul.hMul Bg (Real.exp (HMul.h …
    ⊢ Asymptotics.IsBigO l (HSub.hSub f g) fun z => Real.exp (HMul.hMul (Max.max 0 …
  -/
  refine (hOf.trans_le <| this ?_ ?_ ?_).sub (hOg.trans_le <| this ?_ ?_ ?_)
  exacts [le_max_left _ _, le_max_left _ _, (le_max_left _ _).trans (le_max_right _ _),
    le_max_right _ _, le_max_left _ _, (le_max_right _ _).trans (le_max_right _ _)]


/-- An auxiliary lemma that combines two “exponential of a power” estimates into a similar estimate
on the difference of the functions. -/
theorem isBigO_sub_exp_rpow {a : ℝ} {f g : ℂ → E} {l : Filter ℂ}
    (hBf : ∃ c < a, ∃ B, f =O[cobounded ℂ ⊓ l] fun z => expR (B * abs z ^ c))
    (hBg : ∃ c < a, ∃ B, g =O[cobounded ℂ ⊓ l] fun z => expR (B * abs z ^ c)) :
    ∃ c < a, ∃ B, (f - g) =O[cobounded ℂ ⊓ l] fun z => expR (B * abs z ^ c) := by
  have : ∀ {c₁ c₂ B₁ B₂ : ℝ}, c₁ ≤ c₂ → 0 ≤ B₂ → B₁ ≤ B₂ →
      (fun z : ℂ => expR (B₁ * abs z ^ c₁)) =O[cobounded ℂ ⊓ l]
        fun z => expR (B₂ * abs z ^ c₂) := fun hc hB₀ hB ↦ .of_bound 1 <| by
    filter_upwards [(eventually_cobounded_le_norm 1).filter_mono inf_le_left] with z hz
    simp only [one_mul, Real.norm_eq_abs, Real.abs_exp]
    gcongr; assumption
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : Real
    f g : Complex → E
    l : Filter Complex
    hBf : Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO (Min …
    hBg : Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO (Min …
    this : ∀ {c₁ c₂ B₁ B₂ : Real}, LE.le c₁ c₂ → LE.le 0 B₂ → LE.le B₁ B₂ → Asympt …
    ⊢ Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO (Min.min …
  -/
  rcases hBf with ⟨cf, hcf, Bf, hOf⟩; rcases hBg with ⟨cg, hcg, Bg, hOg⟩
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : Real
    f g : Complex → E
    l : Filter Complex
    this : ∀ {c₁ c₂ B₁ B₂ : Real}, LE.le c₁ c₂ → LE.le 0 B₂ → LE.le B₁ B₂ → Asympt …
    cf : Real
    hcf : LT.lt cf a
    Bf : Real
    hOf : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) l) f fun z =>  …
    cg : Real
    hcg : LT.lt cg a
    Bg : Real
    hOg : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) l) g fun z =>  …
    ⊢ Exists fun c => And (LT.lt c a) (Exists fun B => Asymptotics.IsBigO (Min.min …
  -/
  refine ⟨max cf cg, max_lt hcf hcg, max 0 (max Bf Bg), ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    a : Real
    f g : Complex → E
    l : Filter Complex
    this : ∀ {c₁ c₂ B₁ B₂ : Real}, LE.le c₁ c₂ → LE.le 0 B₂ → LE.le B₁ B₂ → Asympt …
    cf : Real
    hcf : LT.lt cf a
    Bf : Real
    hOf : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) l) f fun z =>  …
    cg : Real
    hcg : LT.lt cg a
    Bg : Real
    hOg : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) l) g fun z =>  …
    ⊢ Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) l) (HSub.hSub f g) …
  -/
  refine (hOf.trans <| this ?_ ?_ ?_).sub (hOg.trans <| this ?_ ?_ ?_)
  exacts [le_max_left _ _, le_max_left _ _, (le_max_left _ _).trans (le_max_right _ _),
    le_max_right _ _, le_max_left _ _, (le_max_right _ _).trans (le_max_right _ _)]


/-- **Phragmen-Lindelöf principle** in a strip `U = {z : ℂ | a < im z < b}`.
Let `f : ℂ → E` be a function such that

* `f` is differentiable on `U` and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * exp(c * |re z|))` on `U` for some `c < π / (b - a)`;
* `‖f z‖` is bounded from above by a constant `C` on the boundary of `U`.

Then `‖f z‖` is bounded by the same constant on the closed strip
`{z : ℂ | a ≤ im z ≤ b}`. Moreover, it suffices to verify the second assumption
only for sufficiently large values of `|re z|`.
-/
theorem horizontal_strip (hfd : DiffContOnCl ℂ f (im ⁻¹' Ioo a b))
    (hB : ∃ c < π / (b - a), ∃ B, f =O[comap (_root_.abs ∘ re) atTop ⊓ 𝓟 (im ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.re|)))
    (hle_a : ∀ z : ℂ, im z = a → ‖f z‖ ≤ C) (hle_b : ∀ z, im z = b → ‖f z‖ ≤ C) (hza : a ≤ im z)
    (hzb : im z ≤ b) : ‖f z‖ ≤ C := by
  -- If `im z = a` or `im z = b`, then we apply `hle_a` or `hle_b`, otherwise `im z ∈ Ioo a b`.
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
    hza : LE.le a z.im
    hzb : LE.le z.im b
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rw [le_iff_eq_or_lt] at hza hzb
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
    hza : Or (Eq a z.im) (LT.lt a z.im)
    hzb : Or (Eq z.im b) (LT.lt z.im b)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  cases' hza with hza hza; · exact hle_a _ hza.symm
                             /-
                               🎉 no goals
                             -/
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
    hzb : Or (Eq z.im b) (LT.lt z.im b)
    hza : LT.lt a z.im
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  cases' hzb with hzb hzb; · exact hle_b _ hzb
                             /-
                               🎉 no goals
                             -/
  /-
    case inr.inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
    hza : LT.lt a z.im
    hzb : LT.lt z.im b
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  wlog hC₀ : 0 < C generalizing C
    /-
      case inr.inr.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      a b C : Real
      f : Complex → E
      z : Complex
      hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
      hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
      hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
      hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
      hza : LT.lt a z.im
      hzb : LT.lt z.im b
      this : ∀ {C : Real}, (∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C)  …
      hC₀ : Not (LT.lt 0 C)
      ⊢ LE.le (Norm.norm (f z)) C
    -/
  · refine le_of_forall_le_of_dense fun C' hC' => this (fun w hw => ?_) (fun w hw => ?_) ?_
      /-
        case inr.inr.inr.refine_1
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        a b C : Real
        f : Complex → E
        z : Complex
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
        hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
        hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
        hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
        hza : LT.lt a z.im
        hzb : LT.lt z.im b
        this : ∀ {C : Real}, (∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C)  …
        hC₀ : Not (LT.lt 0 C)
        C' : Real
        hC' : LT.lt C C'
        w : Complex
        hw : Eq w.im a
        ⊢ LE.le (Norm.norm (f w)) C'
      -/
    · exact (hle_a _ hw).trans hC'.le
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.inr.refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        a b C : Real
        f : Complex → E
        z : Complex
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
        hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
        hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
        hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
        hza : LT.lt a z.im
        hzb : LT.lt z.im b
        this : ∀ {C : Real}, (∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C)  …
        hC₀ : Not (LT.lt 0 C)
        C' : Real
        hC' : LT.lt C C'
        w : Complex
        hw : Eq w.im b
        ⊢ LE.le (Norm.norm (f w)) C'
      -/
    · exact (hle_b _ hw).trans hC'.le
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.inr.refine_3
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        a b C : Real
        f : Complex → E
        z : Complex
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
        hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
        hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
        hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
        hza : LT.lt a z.im
        hzb : LT.lt z.im b
        this : ∀ {C : Real}, (∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C)  …
        hC₀ : Not (LT.lt 0 C)
        C' : Real
        hC' : LT.lt C C'
        ⊢ LT.lt 0 C'
      -/
    · refine ((norm_nonneg (f (a * I))).trans (hle_a _ ?_)).trans_lt hC'
      /-
        case inr.inr.inr.refine_3
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        a b C : Real
        f : Complex → E
        z : Complex
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo a b))
        hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
        hle_a : ∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C
        hle_b : ∀ (z : Complex), Eq z.im b → LE.le (Norm.norm (f z)) C
        hza : LT.lt a z.im
        hzb : LT.lt z.im b
        this : ∀ {C : Real}, (∀ (z : Complex), Eq z.im a → LE.le (Norm.norm (f z)) C)  …
        hC₀ : Not (LT.lt 0 C)
        C' : Real
        hC' : LT.lt C C'
        ⊢ Eq (HMul.hMul (↑a) Complex.I).im a
      -/
      rw [mul_I_im, ofReal_re]
      /-
        🎉 no goals
      -/
  -- After a change of variables, we deal with the strip `a - b < im z < a + b` instead
  -- of `a < im z < b`
  obtain ⟨a, b, rfl, rfl⟩ : ∃ a' b', a = a' - b' ∧ b = a' + b' :=
    ⟨(a + b) / 2, (b - a) / 2, by ring, by ring⟩
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub (HAdd.hAdd a b …
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have hab : a - b < a + b := hza.trans hzb
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub (HAdd.hAdd a b …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have hb : 0 < b := by simpa only [sub_eq_add_neg, add_lt_add_iff_left, neg_lt_self_iff] using hab
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub (HAdd.hAdd a b …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rw [add_sub_sub_cancel, ← two_mul, div_mul_eq_div_div] at hB
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)) (Exists …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have hπb : 0 < π / 2 / b := div_pos Real.pi_div_two_pos hb
  -- Choose some `c B : ℝ` satisfying `hB`, then choose `max c 0 < d < π / 2 / b`.
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)) (Exists …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rcases hB with ⟨c, hc, B, hO⟩
  obtain ⟨d, ⟨hcd, hd₀⟩, hd⟩ : ∃ d, (c < d ∧ 0 < d) ∧ d < π / 2 / b := by
    simpa only [max_lt_iff] using exists_between (max_lt hc hπb)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    c : Real
    hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    B : Real
    hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
    d : Real
    hd : LT.lt d (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    hcd : LT.lt c d
    hd₀ : LT.lt 0 d
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  have hb' : d * b < π / 2 := (lt_div_iff₀ hb).1 hd
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    c : Real
    hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    B : Real
    hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
    d : Real
    hd : LT.lt d (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    hcd : LT.lt c d
    hd₀ : LT.lt 0 d
    hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  set aff := (fun w => d * (w - a * I) : ℂ → ℂ)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    c : Real
    hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    B : Real
    hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
    d : Real
    hd : LT.lt d (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    hcd : LT.lt c d
    hd₀ : LT.lt 0 d
    hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
    aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  set g := fun (ε : ℝ) (w : ℂ) => exp (ε * (exp (aff w) + exp (-aff w)))
  /- Since `g ε z → 1` as `ε → 0⁻`, it suffices to prove that `‖g ε z • f z‖ ≤ C`
    for all negative `ε`. -/
  suffices ∀ᶠ ε : ℝ in 𝓝[<] (0 : ℝ), ‖g ε z • f z‖ ≤ C by
    refine le_of_tendsto (Tendsto.mono_left ?_ nhdsWithin_le_nhds) this
    apply ((continuous_ofReal.mul continuous_const).cexp.smul continuous_const).norm.tendsto'
    simp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    c : Real
    hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    B : Real
    hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
    d : Real
    hd : LT.lt d (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    hcd : LT.lt c d
    hd₀ : LT.lt 0 d
    hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
    aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
    g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
    ⊢ Filter.Eventually (fun ε => LE.le (Norm.norm (HSMul.hSMul (g ε z) (f z))) C) …
  -/
  filter_upwards [self_mem_nhdsWithin] with ε ε₀; change ε < 0 at ε₀
  -- An upper estimate on `‖g ε w‖` that will be used in two branches of the proof.
  obtain ⟨δ, δ₀, hδ⟩ :
    ∃ δ : ℝ,
      δ < 0 ∧ ∀ ⦃w⦄, im w ∈ Icc (a - b) (a + b) → abs (g ε w) ≤ expR (δ * expR (d * |re w|)) := by
    refine
      ⟨ε * Real.cos (d * b),
        mul_neg_of_neg_of_pos ε₀
          (Real.cos_pos_of_mem_Ioo <| abs_lt.1 <| (abs_of_pos (mul_pos hd₀ hb)).symm ▸ hb'),
        fun w hw => ?_⟩
    replace hw : |im (aff w)| ≤ d * b := by
      rw [← Real.closedBall_eq_Icc] at hw
      rwa [im_ofReal_mul, sub_im, mul_I_im, ofReal_re, _root_.abs_mul, abs_of_pos hd₀,
        mul_le_mul_left hd₀]
    simpa only [aff, re_ofReal_mul, _root_.abs_mul, abs_of_pos hd₀, sub_re, mul_I_re, ofReal_im,
      zero_mul, neg_zero, sub_zero] using
      abs_exp_mul_exp_add_exp_neg_le_of_abs_im_le ε₀.le hw hb'.le
  -- `abs (g ε w) ≤ 1` on the lines `w.im = a ± b` (actually, it holds everywhere in the strip)
  have hg₁ : ∀ w, im w = a - b ∨ im w = a + b → abs (g ε w) ≤ 1 := by
    refine fun w hw => (hδ <| hw.by_cases ?_ ?_).trans (Real.exp_le_one_iff.2 ?_)
    exacts [fun h => h.symm ▸ left_mem_Icc.2 hab.le, fun h => h.symm ▸ right_mem_Icc.2 hab.le,
      mul_nonpos_of_nonpos_of_nonneg δ₀.le (Real.exp_pos _).le]
  /- Our apriori estimate on `f` implies that `g ε w • f w → 0` as `|w.re| → ∞` along the strip. In
    particular, its norm is less than or equal to `C` for sufficiently large `|w.re|`. -/
  obtain ⟨R, hzR, hR⟩ :
    ∃ R : ℝ, |z.re| < R ∧ ∀ w, |re w| = R → im w ∈ Ioo (a - b) (a + b) → ‖g ε w • f w‖ ≤ C := by
    refine ((eventually_gt_atTop _).and ?_).exists
    rcases hO.exists_pos with ⟨A, hA₀, hA⟩
    simp only [isBigOWith_iff, eventually_inf_principal, eventually_comap, mem_Ioo, ← abs_lt,
      mem_preimage, (· ∘ ·), Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)] at hA
    suffices
        Tendsto (fun R => expR (δ * expR (d * R) + B * expR (c * R) + Real.log A)) atTop (𝓝 0) by
      filter_upwards [this.eventually (ge_mem_nhds hC₀), hA] with R hR Hle w hre him
      calc
        ‖g ε w • f w‖ ≤ expR (δ * expR (d * R) + B * expR (c * R) + Real.log A) := ?_
        _ ≤ C := hR
      rw [norm_smul, Real.exp_add, ← hre, Real.exp_add, Real.exp_log hA₀, mul_assoc, mul_comm _ A]
      gcongr
      exacts [hδ <| Ioo_subset_Icc_self him, Hle _ hre him]
    refine Real.tendsto_exp_atBot.comp ?_
    suffices H : Tendsto (fun R => δ + B * (expR ((d - c) * R))⁻¹) atTop (𝓝 (δ + B * 0)) by
      rw [mul_zero, add_zero] at H
      refine Tendsto.atBot_add ?_ tendsto_const_nhds
      simpa only [id, (· ∘ ·), add_mul, mul_assoc, ← div_eq_inv_mul, ← Real.exp_sub, ← sub_mul,
        sub_sub_cancel]
        using H.neg_mul_atTop δ₀ <| Real.tendsto_exp_atTop.comp <|
          tendsto_const_nhds.mul_atTop hd₀ tendsto_id
    refine tendsto_const_nhds.add (tendsto_const_nhds.mul ?_)
    exact tendsto_inv_atTop_zero.comp <| Real.tendsto_exp_atTop.comp <|
      tendsto_const_nhds.mul_atTop (sub_pos.2 hcd) tendsto_id
  /-
    case h.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C✝ : Real
    f : Complex → E
    z : Complex
    C : Real
    hC₀ : LT.lt 0 C
    a b : Real
    hza : LT.lt (HSub.hSub a b) z.im
    hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
    hzb : LT.lt z.im (HAdd.hAdd a b)
    hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
    hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
    hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
    hb : LT.lt 0 b
    hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    c : Real
    hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    B : Real
    hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
    d : Real
    hd : LT.lt d (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
    hcd : LT.lt c d
    hd₀ : LT.lt 0 d
    hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
    aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
    g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
    ε : Real
    ε₀ : LT.lt ε 0
    δ : Real
    δ₀ : LT.lt δ 0
    hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
    hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
    R : Real
    hzR : LT.lt (abs z.re) R
    hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
    ⊢ LE.le (Norm.norm (HSMul.hSMul (g ε z) (f z))) C
  -/
  have hR₀ : 0 < R := (_root_.abs_nonneg _).trans_lt hzR
  /- Finally, we apply the bounded version of the maximum modulus principle to the rectangle
    `(-R, R) × (a - b, a + b)`. The function is bounded by `C` on the horizontal sides by assumption
    (and because `‖g ε w‖ ≤ 1`) and on the vertical sides by the choice of `R`. -/
  have hgd : Differentiable ℂ (g ε) :=
    ((((differentiable_id.sub_const _).const_mul _).cexp.add
            ((differentiable_id.sub_const _).const_mul _).neg.cexp).const_mul _).cexp
  replace hd : DiffContOnCl ℂ (fun w => g ε w • f w) (Ioo (-R) R ×ℂ Ioo (a - b) (a + b)) :=
    (hgd.diffContOnCl.smul hfd).mono inter_subset_right
  convert norm_le_of_forall_mem_frontier_norm_le ((isBounded_Ioo _ _).reProdIm (isBounded_Ioo _ _))
    hd (fun w hw => _) _
  · rw [frontier_reProdIm, closure_Ioo (neg_lt_self hR₀).ne, frontier_Ioo hab, closure_Ioo hab.ne,
      frontier_Ioo (neg_lt_self hR₀)] at hw
    /-
      case h.intro.intro.intro.intro.convert_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C✝ : Real
      f : Complex → E
      z : Complex
      C : Real
      hC₀ : LT.lt 0 C
      a b : Real
      hza : LT.lt (HSub.hSub a b) z.im
      hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
      hzb : LT.lt z.im (HAdd.hAdd a b)
      hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
      hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
      hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
      hb : LT.lt 0 b
      hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
      c : Real
      hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
      d : Real
      hcd : LT.lt c d
      hd₀ : LT.lt 0 d
      hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
      aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
      g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
      ε : Real
      ε₀ : LT.lt ε 0
      δ : Real
      δ₀ : LT.lt δ 0
      hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
      hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
      R : Real
      hzR : LT.lt (abs z.re) R
      hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
      hR₀ : LT.lt 0 R
      hgd : Differentiable Complex (g ε)
      hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
      w : Complex
      hw : Membership.mem (Union.union (Complex.reProdIm (Set.Icc (Neg.neg R) R) (In …
      ⊢ LE.le (Norm.norm (HSMul.hSMul (g ε w) (f w))) C
    -/
    by_cases him : w.im = a - b ∨ w.im = a + b
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C✝ : Real
        f : Complex → E
        z : Complex
        C : Real
        hC₀ : LT.lt 0 C
        a b : Real
        hza : LT.lt (HSub.hSub a b) z.im
        hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
        hzb : LT.lt z.im (HAdd.hAdd a b)
        hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
        hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
        hb : LT.lt 0 b
        hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        c : Real
        hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
        d : Real
        hcd : LT.lt c d
        hd₀ : LT.lt 0 d
        hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
        aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
        g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
        ε : Real
        ε₀ : LT.lt ε 0
        δ : Real
        δ₀ : LT.lt δ 0
        hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
        hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
        R : Real
        hzR : LT.lt (abs z.re) R
        hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
        hR₀ : LT.lt 0 R
        hgd : Differentiable Complex (g ε)
        hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
        w : Complex
        hw : Membership.mem (Union.union (Complex.reProdIm (Set.Icc (Neg.neg R) R) (In …
        him : Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))
        ⊢ LE.le (Norm.norm (HSMul.hSMul (g ε w) (f w))) C
      -/
    · rw [norm_smul, ← one_mul C]
      /-
        case pos
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C✝ : Real
        f : Complex → E
        z : Complex
        C : Real
        hC₀ : LT.lt 0 C
        a b : Real
        hza : LT.lt (HSub.hSub a b) z.im
        hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
        hzb : LT.lt z.im (HAdd.hAdd a b)
        hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
        hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
        hb : LT.lt 0 b
        hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        c : Real
        hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
        d : Real
        hcd : LT.lt c d
        hd₀ : LT.lt 0 d
        hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
        aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
        g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
        ε : Real
        ε₀ : LT.lt ε 0
        δ : Real
        δ₀ : LT.lt δ 0
        hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
        hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
        R : Real
        hzR : LT.lt (abs z.re) R
        hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
        hR₀ : LT.lt 0 R
        hgd : Differentiable Complex (g ε)
        hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
        w : Complex
        hw : Membership.mem (Union.union (Complex.reProdIm (Set.Icc (Neg.neg R) R) (In …
        him : Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))
        ⊢ LE.le (HMul.hMul (Norm.norm (g ε w)) (Norm.norm (f w))) (HMul.hMul 1 C)
      -/
      exact mul_le_mul (hg₁ _ him) (him.by_cases (hle_a _) (hle_b _)) (norm_nonneg _) zero_le_one
      /-
        🎉 no goals
      -/
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C✝ : Real
        f : Complex → E
        z : Complex
        C : Real
        hC₀ : LT.lt 0 C
        a b : Real
        hza : LT.lt (HSub.hSub a b) z.im
        hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
        hzb : LT.lt z.im (HAdd.hAdd a b)
        hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
        hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
        hb : LT.lt 0 b
        hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        c : Real
        hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
        d : Real
        hcd : LT.lt c d
        hd₀ : LT.lt 0 d
        hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
        aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
        g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
        ε : Real
        ε₀ : LT.lt ε 0
        δ : Real
        δ₀ : LT.lt δ 0
        hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
        hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
        R : Real
        hzR : LT.lt (abs z.re) R
        hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
        hR₀ : LT.lt 0 R
        hgd : Differentiable Complex (g ε)
        hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
        w : Complex
        hw : Membership.mem (Union.union (Complex.reProdIm (Set.Icc (Neg.neg R) R) (In …
        him : Not (Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b)))
        ⊢ LE.le (Norm.norm (HSMul.hSMul (g ε w) (f w))) C
      -/
    · replace hw : w ∈ {-R, R} ×ℂ Icc (a - b) (a + b) := hw.resolve_left fun h ↦ him h.2
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C✝ : Real
        f : Complex → E
        z : Complex
        C : Real
        hC₀ : LT.lt 0 C
        a b : Real
        hza : LT.lt (HSub.hSub a b) z.im
        hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
        hzb : LT.lt z.im (HAdd.hAdd a b)
        hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
        hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
        hb : LT.lt 0 b
        hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        c : Real
        hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
        d : Real
        hcd : LT.lt c d
        hd₀ : LT.lt 0 d
        hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
        aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
        g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
        ε : Real
        ε₀ : LT.lt ε 0
        δ : Real
        δ₀ : LT.lt δ 0
        hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
        hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
        R : Real
        hzR : LT.lt (abs z.re) R
        hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
        hR₀ : LT.lt 0 R
        hgd : Differentiable Complex (g ε)
        hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
        w : Complex
        him : Not (Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b)))
        hw : Membership.mem (Complex.reProdIm (Insert.insert (Neg.neg R) (Singleton.si …
        ⊢ LE.le (Norm.norm (HSMul.hSMul (g ε w) (f w))) C
      -/
      have hw' := eq_endpoints_or_mem_Ioo_of_mem_Icc hw.2; rw [← or_assoc] at hw'
      /-
        case neg
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C✝ : Real
        f : Complex → E
        z : Complex
        C : Real
        hC₀ : LT.lt 0 C
        a b : Real
        hza : LT.lt (HSub.hSub a b) z.im
        hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
        hzb : LT.lt z.im (HAdd.hAdd a b)
        hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
        hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
        hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
        hb : LT.lt 0 b
        hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        c : Real
        hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
        d : Real
        hcd : LT.lt c d
        hd₀ : LT.lt 0 d
        hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
        aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
        g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
        ε : Real
        ε₀ : LT.lt ε 0
        δ : Real
        δ₀ : LT.lt δ 0
        hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
        hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
        R : Real
        hzR : LT.lt (abs z.re) R
        hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
        hR₀ : LT.lt 0 R
        hgd : Differentiable Complex (g ε)
        hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
        w : Complex
        him : Not (Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b)))
        hw : Membership.mem (Complex.reProdIm (Insert.insert (Neg.neg R) (Singleton.si …
        hw' : Or (Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))) (Membership. …
        ⊢ LE.le (Norm.norm (HSMul.hSMul (g ε w) (f w))) C
      -/
      exact hR _ ((abs_eq hR₀.le).2 hw.1.symm) (hw'.resolve_left him)
      /-
        🎉 no goals
      -/
    /-
      case h.intro.intro.intro.intro.convert_4
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C✝ : Real
      f : Complex → E
      z : Complex
      C : Real
      hC₀ : LT.lt 0 C
      a b : Real
      hza : LT.lt (HSub.hSub a b) z.im
      hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
      hzb : LT.lt z.im (HAdd.hAdd a b)
      hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
      hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
      hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
      hb : LT.lt 0 b
      hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
      c : Real
      hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
      d : Real
      hcd : LT.lt c d
      hd₀ : LT.lt 0 d
      hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
      aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
      g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
      ε : Real
      ε₀ : LT.lt ε 0
      δ : Real
      δ₀ : LT.lt δ 0
      hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
      hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
      R : Real
      hzR : LT.lt (abs z.re) R
      hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
      hR₀ : LT.lt 0 R
      hgd : Differentiable Complex (g ε)
      hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
      ⊢ Membership.mem (closure (Complex.reProdIm (Set.Ioo (Neg.neg R) R) (Set.Ioo ( …
    -/
  · rw [closure_reProdIm, closure_Ioo hab.ne, closure_Ioo (neg_lt_self hR₀).ne]
    /-
      case h.intro.intro.intro.intro.convert_4
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C✝ : Real
      f : Complex → E
      z : Complex
      C : Real
      hC₀ : LT.lt 0 C
      a b : Real
      hza : LT.lt (HSub.hSub a b) z.im
      hle_a : ∀ (z : Complex), Eq z.im (HSub.hSub a b) → LE.le (Norm.norm (f z)) C
      hzb : LT.lt z.im (HAdd.hAdd a b)
      hle_b : ∀ (z : Complex), Eq z.im (HAdd.hAdd a b) → LE.le (Norm.norm (f z)) C
      hfd : DiffContOnCl Complex f (Set.preimage Complex.im (Set.Ioo (HSub.hSub a b) …
      hab : LT.lt (HSub.hSub a b) (HAdd.hAdd a b)
      hb : LT.lt 0 b
      hπb : LT.lt 0 (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
      c : Real
      hc : LT.lt c (HDiv.hDiv (HDiv.hDiv Real.pi 2) b)
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re)  …
      d : Real
      hcd : LT.lt c d
      hd₀ : LT.lt 0 d
      hb' : LT.lt (HMul.hMul d b) (HDiv.hDiv Real.pi 2)
      aff : Complex → Complex := fun w => HMul.hMul (↑d) (HSub.hSub w (HMul.hMul (↑a …
      g : Real → Complex → Complex := fun ε w => Complex.exp (HMul.hMul (↑ε) (HAdd.h …
      ε : Real
      ε₀ : LT.lt ε 0
      δ : Real
      δ₀ : LT.lt δ 0
      hδ : ∀ ⦃w : Complex⦄, Membership.mem (Set.Icc (HSub.hSub a b) (HAdd.hAdd a b)) …
      hg₁ : ∀ (w : Complex), Or (Eq w.im (HSub.hSub a b)) (Eq w.im (HAdd.hAdd a b))  …
      R : Real
      hzR : LT.lt (abs z.re) R
      hR : ∀ (w : Complex), Eq (abs w.re) R → Membership.mem (Set.Ioo (HSub.hSub a b …
      hR₀ : LT.lt 0 R
      hgd : Differentiable Complex (g ε)
      hd : DiffContOnCl Complex (fun w => HSMul.hSMul (g ε w) (f w)) (Complex.reProd …
      ⊢ Membership.mem (Complex.reProdIm (Set.Icc (Neg.neg R) R) (Set.Icc (HSub.hSub …
    -/
    exact ⟨abs_le.1 hzR.le, ⟨hza.le, hzb.le⟩⟩
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in a strip `U = {z : ℂ | a < im z < b}`.
Let `f : ℂ → E` be a function such that

* `f` is differentiable on `U` and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * exp(c * |re z|))` on `U` for some `c < π / (b - a)`;
* `f z = 0` on the boundary of `U`.

Then `f` is equal to zero on the closed strip `{z : ℂ | a ≤ im z ≤ b}`.
-/
theorem eq_zero_on_horizontal_strip (hd : DiffContOnCl ℂ f (im ⁻¹' Ioo a b))
    (hB : ∃ c < π / (b - a), ∃ B, f =O[comap (_root_.abs ∘ re) atTop ⊓ 𝓟 (im ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.re|)))
    (ha : ∀ z : ℂ, z.im = a → f z = 0) (hb : ∀ z : ℂ, z.im = b → f z = 0) :
    EqOn f 0 (im ⁻¹' Icc a b) := fun _z hz =>
  norm_le_zero_iff.1 <| horizontal_strip hd hB (fun z hz => (ha z hz).symm ▸ norm_zero.le)
    (fun z hz => (hb z hz).symm ▸ norm_zero.le) hz.1 hz.2


/-- **Phragmen-Lindelöf principle** in a strip `U = {z : ℂ | a < im z < b}`.
Let `f g : ℂ → E` be functions such that

* `f` and `g` are differentiable on `U` and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * exp(c * |re z|))` on `U` for some
  `c < π / (b - a)`;
* `f z = g z` on the boundary of `U`.

Then `f` is equal to `g` on the closed strip `{z : ℂ | a ≤ im z ≤ b}`.
-/
theorem eqOn_horizontal_strip {g : ℂ → E} (hdf : DiffContOnCl ℂ f (im ⁻¹' Ioo a b))
    (hBf : ∃ c < π / (b - a), ∃ B, f =O[comap (_root_.abs ∘ re) atTop ⊓ 𝓟 (im ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.re|)))
    (hdg : DiffContOnCl ℂ g (im ⁻¹' Ioo a b))
    (hBg : ∃ c < π / (b - a), ∃ B, g =O[comap (_root_.abs ∘ re) atTop ⊓ 𝓟 (im ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.re|)))
    (ha : ∀ z : ℂ, z.im = a → f z = g z) (hb : ∀ z : ℂ, z.im = b → f z = g z) :
    EqOn f g (im ⁻¹' Icc a b) := fun _z hz =>
  sub_eq_zero.1 (eq_zero_on_horizontal_strip (hdf.sub hdg) (isBigO_sub_exp_exp hBf hBg)
    (fun w hw => sub_eq_zero.2 (ha w hw)) (fun w hw => sub_eq_zero.2 (hb w hw)) hz)


/-- **Phragmen-Lindelöf principle** in a strip `U = {z : ℂ | a < re z < b}`.
Let `f : ℂ → E` be a function such that

* `f` is differentiable on `U` and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * exp(c * |im z|))` on `U` for some `c < π / (b - a)`;
* `‖f z‖` is bounded from above by a constant `C` on the boundary of `U`.

Then `‖f z‖` is bounded by the same constant on the closed strip
`{z : ℂ | a ≤ re z ≤ b}`. Moreover, it suffices to verify the second assumption
only for sufficiently large values of `|im z|`.
-/
theorem vertical_strip (hfd : DiffContOnCl ℂ f (re ⁻¹' Ioo a b))
    (hB : ∃ c < π / (b - a), ∃ B, f =O[comap (_root_.abs ∘ im) atTop ⊓ 𝓟 (re ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.im|)))
    (hle_a : ∀ z : ℂ, re z = a → ‖f z‖ ≤ C) (hle_b : ∀ z, re z = b → ‖f z‖ ≤ C) (hza : a ≤ re z)
    (hzb : re z ≤ b) : ‖f z‖ ≤ C := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.re (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.re a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.re b → LE.le (Norm.norm (f z)) C
    hza : LE.le a z.re
    hzb : LE.le z.re b
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  suffices ‖f (z * I * -I)‖ ≤ C by simpa [mul_assoc] using this
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.re (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.re a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.re b → LE.le (Norm.norm (f z)) C
    hza : LE.le a z.re
    hzb : LE.le z.re b
    ⊢ LE.le (Norm.norm (f (HMul.hMul (HMul.hMul z Complex.I) (Neg.neg Complex.I))) …
  -/
  have H : MapsTo (· * -I) (im ⁻¹' Ioo a b) (re ⁻¹' Ioo a b) := fun z hz ↦ by simpa using hz
  refine horizontal_strip (f := fun z ↦ f (z * -I))
    (hfd.comp (differentiable_id.mul_const _).diffContOnCl H) ?_ (fun z hz => hle_a _ ?_)
    (fun z hz => hle_b _ ?_) ?_ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      a b C : Real
      f : Complex → E
      z : Complex
      hfd : DiffContOnCl Complex f (Set.preimage Complex.re (Set.Ioo a b))
      hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
      hle_a : ∀ (z : Complex), Eq z.re a → LE.le (Norm.norm (f z)) C
      hle_b : ∀ (z : Complex), Eq z.re b → LE.le (Norm.norm (f z)) C
      hza : LE.le a z.re
      hzb : LE.le z.re b
      H : Set.MapsTo (fun x => HMul.hMul x (Neg.neg Complex.I)) (Set.preimage Comple …
      ⊢ Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists fu …
    -/
  · rcases hB with ⟨c, hc, B, hO⟩
    /-
      case refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      a b C : Real
      f : Complex → E
      z : Complex
      hfd : DiffContOnCl Complex f (Set.preimage Complex.re (Set.Ioo a b))
      hle_a : ∀ (z : Complex), Eq z.re a → LE.le (Norm.norm (f z)) C
      hle_b : ∀ (z : Complex), Eq z.re b → LE.le (Norm.norm (f z)) C
      hza : LE.le a z.re
      hzb : LE.le z.re b
      H : Set.MapsTo (fun x => HMul.hMul x (Neg.neg Complex.I)) (Set.preimage Comple …
      c : Real
      hc : LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.im)  …
      ⊢ Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists fu …
    -/
    refine ⟨c, hc, B, ?_⟩
    have : Tendsto (· * -I) (comap (|re ·|) atTop ⊓ 𝓟 (im ⁻¹' Ioo a b))
        (comap (|im ·|) atTop ⊓ 𝓟 (re ⁻¹' Ioo a b)) := by
      refine (tendsto_comap_iff.2 ?_).inf H.tendsto
      simpa [Function.comp_def] using tendsto_comap
    /-
      case refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      a b C : Real
      f : Complex → E
      z : Complex
      hfd : DiffContOnCl Complex f (Set.preimage Complex.re (Set.Ioo a b))
      hle_a : ∀ (z : Complex), Eq z.re a → LE.le (Norm.norm (f z)) C
      hle_b : ∀ (z : Complex), Eq z.re b → LE.le (Norm.norm (f z)) C
      hza : LE.le a z.re
      hzb : LE.le z.re b
      H : Set.MapsTo (fun x => HMul.hMul x (Neg.neg Complex.I)) (Set.preimage Comple …
      c : Real
      hc : LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.im)  …
      this : Filter.Tendsto (fun x => HMul.hMul x (Neg.neg Complex.I)) (Min.min (Fil …
      ⊢ Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re) Fil …
    -/
    simpa [Function.comp_def] using hO.comp_tendsto this
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    a b C : Real
    f : Complex → E
    z✝ : Complex
    hfd : DiffContOnCl Complex f (Set.preimage Complex.re (Set.Ioo a b))
    hB : Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub b a))) (Exists …
    hle_a : ∀ (z : Complex), Eq z.re a → LE.le (Norm.norm (f z)) C
    hle_b : ∀ (z : Complex), Eq z.re b → LE.le (Norm.norm (f z)) C
    hza : LE.le a z✝.re
    hzb : LE.le z✝.re b
    H : Set.MapsTo (fun x => HMul.hMul x (Neg.neg Complex.I)) (Set.preimage Comple …
    z : Complex
    hz : Eq z.im a
    ⊢ Eq (HMul.hMul z (Neg.neg Complex.I)).re a
  -/
  all_goals simpa
  /-
    🎉 no goals
  -/


/-- **Phragmen-Lindelöf principle** in a strip `U = {z : ℂ | a < re z < b}`.
Let `f : ℂ → E` be a function such that

* `f` is differentiable on `U` and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * exp(c * |im z|))` on `U` for some `c < π / (b - a)`;
* `f z = 0` on the boundary of `U`.

Then `f` is equal to zero on the closed strip `{z : ℂ | a ≤ re z ≤ b}`.
-/
theorem eq_zero_on_vertical_strip (hd : DiffContOnCl ℂ f (re ⁻¹' Ioo a b))
    (hB : ∃ c < π / (b - a), ∃ B, f =O[comap (_root_.abs ∘ im) atTop ⊓ 𝓟 (re ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.im|)))
    (ha : ∀ z : ℂ, re z = a → f z = 0) (hb : ∀ z : ℂ, re z = b → f z = 0) :
    EqOn f 0 (re ⁻¹' Icc a b) := fun _z hz =>
  norm_le_zero_iff.1 <| vertical_strip hd hB (fun z hz => (ha z hz).symm ▸ norm_zero.le)
    (fun z hz => (hb z hz).symm ▸ norm_zero.le) hz.1 hz.2


/-- **Phragmen-Lindelöf principle** in a strip `U = {z : ℂ | a < re z < b}`.
Let `f g : ℂ → E` be functions such that

* `f` and `g` are differentiable on `U` and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * exp(c * |im z|))` on `U` for some
  `c < π / (b - a)`;
* `f z = g z` on the boundary of `U`.

Then `f` is equal to `g` on the closed strip `{z : ℂ | a ≤ re z ≤ b}`.
-/
theorem eqOn_vertical_strip {g : ℂ → E} (hdf : DiffContOnCl ℂ f (re ⁻¹' Ioo a b))
    (hBf : ∃ c < π / (b - a), ∃ B, f =O[comap (_root_.abs ∘ im) atTop ⊓ 𝓟 (re ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.im|)))
    (hdg : DiffContOnCl ℂ g (re ⁻¹' Ioo a b))
    (hBg : ∃ c < π / (b - a), ∃ B, g =O[comap (_root_.abs ∘ im) atTop ⊓ 𝓟 (re ⁻¹' Ioo a b)]
      fun z ↦ expR (B * expR (c * |z.im|)))
    (ha : ∀ z : ℂ, re z = a → f z = g z) (hb : ∀ z : ℂ, re z = b → f z = g z) :
    EqOn f g (re ⁻¹' Icc a b) := fun _z hz =>
  sub_eq_zero.1 (eq_zero_on_vertical_strip (hdf.sub hdg) (isBigO_sub_exp_exp hBf hBg)
    (fun w hw => sub_eq_zero.2 (ha w hw)) (fun w hw => sub_eq_zero.2 (hb w hw)) hz)


/-- **Phragmen-Lindelöf principle** in the first quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open first quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open first quadrant
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant `C` on the boundary of the first quadrant.

Then `‖f z‖` is bounded from above by the same constant on the closed first quadrant. -/
nonrec theorem quadrant_I (hd : DiffContOnCl ℂ f (Ioi 0 ×ℂ Ioi 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, 0 ≤ x → ‖f x‖ ≤ C) (him : ∀ x : ℝ, 0 ≤ x → ‖f (x * I)‖ ≤ C) (hz_re : 0 ≤ z.re)
    (hz_im : 0 ≤ z.im) : ‖f z‖ ≤ C := by
  -- The case `z = 0` is trivial.
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    hz_re : LE.le 0 z.re
    hz_im : LE.le 0 z.im
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  rcases eq_or_ne z 0 with (rfl | hzne)
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      hz_re : LE.le 0 (Complex.re 0)
      hz_im : LE.le 0 (Complex.im 0)
      ⊢ LE.le (Norm.norm (f 0)) C
    -/
  · exact hre 0 le_rfl
    /-
      🎉 no goals
    -/
  -- Otherwise, `z = e ^ ζ` for some `ζ : ℂ`, `0 < Im ζ < π / 2`.
  obtain ⟨ζ, hζ, rfl⟩ : ∃ ζ : ℂ, ζ.im ∈ Icc 0 (π / 2) ∧ exp ζ = z := by
    refine ⟨log z, ?_, exp_log hzne⟩
    rw [log_im]
    exact ⟨arg_nonneg_iff.2 hz_im, arg_le_pi_div_two_iff.2 (Or.inl hz_re)⟩
  -- Porting note: failed to clear `clear hz_re hz_im hzne`
  -- We are going to apply `PhragmenLindelof.horizontal_strip` to `f ∘ Complex.exp` and `ζ`.
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    ζ : Complex
    hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
    hz_re : LE.le 0 (Complex.exp ζ).re
    hz_im : LE.le 0 (Complex.exp ζ).im
    hzne : Ne (Complex.exp ζ) 0
    ⊢ LE.le (Norm.norm (f (Complex.exp ζ))) C
  -/
  change ‖(f ∘ exp) ζ‖ ≤ C
  have H : MapsTo exp (im ⁻¹' Ioo 0 (π / 2)) (Ioi 0 ×ℂ Ioi 0) := fun z hz ↦ by
    rw [mem_reProdIm, exp_re, exp_im, mem_Ioi, mem_Ioi]
    have : 0 < Real.cos z.im := Real.cos_pos_of_mem_Ioo ⟨by linarith [hz.1, hz.2], hz.2⟩
    have : 0 < Real.sin z.im :=
      Real.sin_pos_of_mem_Ioo ⟨hz.1, hz.2.trans (half_lt_self Real.pi_pos)⟩
    constructor <;> positivity
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    ζ : Complex
    hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
    hz_re : LE.le 0 (Complex.exp ζ).re
    hz_im : LE.le 0 (Complex.exp ζ).im
    hzne : Ne (Complex.exp ζ) 0
    H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
    ⊢ LE.le (Norm.norm (Function.comp f Complex.exp ζ)) C
  -/
  refine horizontal_strip (hd.comp differentiable_exp.diffContOnCl H) ?_ ?_ ?_ hζ.1 hζ.2
  -- Porting note: failed to clear hζ ζ
  · -- The estimate `hB` on `f` implies the required estimate on
    -- `f ∘ exp` with the same `c` and `B' = max B 0`.
    /-
      case inr.intro.intro.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ : Complex
      hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
      hz_re : LE.le 0 (Complex.exp ζ).re
      hz_im : LE.le 0 (Complex.exp ζ).im
      hzne : Ne (Complex.exp ζ) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ⊢ Exists fun c => And (LT.lt c (HDiv.hDiv Real.pi (HSub.hSub (HDiv.hDiv Real.p …
    -/
    rw [sub_zero, div_div_cancel₀ Real.pi_pos.ne']
    /-
      case inr.intro.intro.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ : Complex
      hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
      hz_re : LE.le 0 (Complex.exp ζ).re
      hz_im : LE.le 0 (Complex.exp ζ).im
      hzne : Ne (Complex.exp ζ) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
    rcases hB with ⟨c, hc, B, hO⟩
    /-
      case inr.intro.intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ : Complex
      hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
      hz_re : LE.le 0 (Complex.exp ζ).re
      hz_im : LE.le 0 (Complex.exp ζ).im
      hzne : Ne (Complex.exp ζ) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
    refine ⟨c, hc, max B 0, ?_⟩
    /-
      case inr.intro.intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ : Complex
      hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
      hz_re : LE.le 0 (Complex.exp ζ).re
      hz_im : LE.le 0 (Complex.exp ζ).im
      hzne : Ne (Complex.exp ζ) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Asymptotics.IsBigO (Min.min (Filter.comap (Function.comp abs Complex.re) Fil …
    -/
    rw [← comap_comap, comap_abs_atTop, comap_sup, inf_sup_right]
    -- We prove separately the estimates as `ζ.re → ∞` and as `ζ.re → -∞`
    refine IsBigO.sup ?_
      ((hO.comp_tendsto <| tendsto_exp_comap_re_atTop.inf H.tendsto).trans <| .of_bound 1 ?_)
    · -- For the estimate as `ζ.re → -∞`, note that `f` is continuous within the first quadrant at
      -- zero, hence `f (exp ζ)` has a limit as `ζ.re → -∞`, `0 < ζ.im < π / 2`.
      have hc : ContinuousWithinAt f (Ioi 0 ×ℂ Ioi 0) 0 := by
        refine (hd.continuousOn _ ?_).mono subset_closure
        simp [closure_reProdIm, mem_reProdIm]
      refine ((hc.tendsto.comp <| tendsto_exp_comap_re_atBot.inf H.tendsto).isBigO_one ℝ).trans
        (isBigO_of_le _ fun w => ?_)
      /-
        case inr.intro.intro.refine_1.intro.intro.intro.refine_1
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C : Real
        f : Complex → E
        hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
        hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
        him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
        ζ : Complex
        hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
        hz_re : LE.le 0 (Complex.exp ζ).re
        hz_im : LE.le 0 (Complex.exp ζ).im
        hzne : Ne (Complex.exp ζ) 0
        H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
        c : Real
        hc✝ : LT.lt c 2
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
        hc : ContinuousWithinAt f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) 0
        w : Complex
        ⊢ LE.le (Norm.norm 1) (Norm.norm (Real.exp (HMul.hMul (Max.max B 0) (Real.exp  …
      -/
      rw [norm_one, Real.norm_of_nonneg (Real.exp_pos _).le, Real.one_le_exp_iff]
      /-
        case inr.intro.intro.refine_1.intro.intro.intro.refine_1
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C : Real
        f : Complex → E
        hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
        hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
        him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
        ζ : Complex
        hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
        hz_re : LE.le 0 (Complex.exp ζ).re
        hz_im : LE.le 0 (Complex.exp ζ).im
        hzne : Ne (Complex.exp ζ) 0
        H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
        c : Real
        hc✝ : LT.lt c 2
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
        hc : ContinuousWithinAt f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) 0
        w : Complex
        ⊢ LE.le 0 (HMul.hMul (Max.max B 0) (Real.exp (HMul.hMul c (abs w.re))))
      -/
      positivity
      /-
        🎉 no goals
      -/
    · -- For the estimate as `ζ.re → ∞`, we reuse the upper estimate on `f`
      simp only [eventually_inf_principal, eventually_comap, comp_apply, one_mul,
        Real.norm_of_nonneg (Real.exp_pos _).le, abs_exp, ← Real.exp_mul, Real.exp_le_exp]
      /-
        case inr.intro.intro.refine_1.intro.intro.intro.refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C : Real
        f : Complex → E
        hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
        hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
        him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
        ζ : Complex
        hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
        hz_re : LE.le 0 (Complex.exp ζ).re
        hz_im : LE.le 0 (Complex.exp ζ).im
        hzne : Ne (Complex.exp ζ) 0
        H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
        c : Real
        hc : LT.lt c 2
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
        ⊢ Filter.Eventually (fun b => ∀ (a : Complex), Eq a.re b → Membership.mem (Set …
      -/
      refine (eventually_ge_atTop 0).mono fun x hx z hz _ => ?_
      /-
        case inr.intro.intro.refine_1.intro.intro.intro.refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C : Real
        f : Complex → E
        hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
        hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
        him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
        ζ : Complex
        hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
        hz_re : LE.le 0 (Complex.exp ζ).re
        hz_im : LE.le 0 (Complex.exp ζ).im
        hzne : Ne (Complex.exp ζ) 0
        H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
        c : Real
        hc : LT.lt c 2
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
        x : Real
        hx : LE.le 0 x
        z : Complex
        hz : Eq z.re x
        x✝ : Membership.mem (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real.pi 2)) …
        ⊢ LE.le (HMul.hMul B (Real.exp (HMul.hMul z.re c))) (HMul.hMul (Max.max B 0) ( …
      -/
      rw [hz, _root_.abs_of_nonneg hx, mul_comm _ c]
      /-
        case inr.intro.intro.refine_1.intro.intro.intro.refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        C : Real
        f : Complex → E
        hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
        hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
        him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
        ζ : Complex
        hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
        hz_re : LE.le 0 (Complex.exp ζ).re
        hz_im : LE.le 0 (Complex.exp ζ).im
        hzne : Ne (Complex.exp ζ) 0
        H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
        c : Real
        hc : LT.lt c 2
        B : Real
        hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
        x : Real
        hx : LE.le 0 x
        z : Complex
        hz : Eq z.re x
        x✝ : Membership.mem (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real.pi 2)) …
        ⊢ LE.le (HMul.hMul B (Real.exp (HMul.hMul c x))) (HMul.hMul (Max.max B 0) (Rea …
      -/
      gcongr; apply le_max_left
              /-
                🎉 no goals
              -/
  · -- If `ζ.im = 0`, then `Complex.exp ζ` is a positive real number
    /-
      case inr.intro.intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ : Complex
      hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
      hz_re : LE.le 0 (Complex.exp ζ).re
      hz_im : LE.le 0 (Complex.exp ζ).im
      hzne : Ne (Complex.exp ζ) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ⊢ ∀ (z : Complex), Eq z.im 0 → LE.le (Norm.norm (Function.comp f Complex.exp z …
    -/
    intro ζ hζ; lift ζ to ℝ using hζ
    /-
      case inr.intro.intro.refine_2.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ✝ : Complex
      hz_re : LE.le 0 (Complex.exp ζ✝).re
      hz_im : LE.le 0 (Complex.exp ζ✝).im
      hzne : Ne (Complex.exp ζ✝) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ζ : Real
      ⊢ LE.le (Norm.norm (Function.comp f Complex.exp ↑ζ)) C
    -/
    rw [comp_apply, ← ofReal_exp]
    /-
      case inr.intro.intro.refine_2.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ✝ : Complex
      hz_re : LE.le 0 (Complex.exp ζ✝).re
      hz_im : LE.le 0 (Complex.exp ζ✝).im
      hzne : Ne (Complex.exp ζ✝) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ζ : Real
      ⊢ LE.le (Norm.norm (f ↑(Real.exp ζ))) C
    -/
    exact hre _ (Real.exp_pos _).le
    /-
      🎉 no goals
    -/
  · -- If `ζ.im = π / 2`, then `Complex.exp ζ` is a purely imaginary number with positive `im`
    /-
      case inr.intro.intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ : Complex
      hζ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ.im
      hz_re : LE.le 0 (Complex.exp ζ).re
      hz_im : LE.le 0 (Complex.exp ζ).im
      hzne : Ne (Complex.exp ζ) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ⊢ ∀ (z : Complex), Eq z.im (HDiv.hDiv Real.pi 2) → LE.le (Norm.norm (Function. …
    -/
    intro ζ hζ
    rw [← re_add_im ζ, hζ, comp_apply, exp_add_mul_I, ← ofReal_cos, ← ofReal_sin,
      Real.cos_pi_div_two, Real.sin_pi_div_two, ofReal_zero, ofReal_one, one_mul, zero_add, ←
      ofReal_exp]
    /-
      case inr.intro.intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      ζ✝ : Complex
      hζ✝ : Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) ζ✝.im
      hz_re : LE.le 0 (Complex.exp ζ✝).re
      hz_im : LE.le 0 (Complex.exp ζ✝).im
      hzne : Ne (Complex.exp ζ✝) 0
      H : Set.MapsTo Complex.exp (Set.preimage Complex.im (Set.Ioo 0 (HDiv.hDiv Real …
      ζ : Complex
      hζ : Eq ζ.im (HDiv.hDiv Real.pi 2)
      ⊢ LE.le (Norm.norm (f (HMul.hMul (↑(Real.exp ζ.re)) Complex.I))) C
    -/
    exact him _ (Real.exp_pos _).le
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the first quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open first quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open first quadrant
  for some `A`, `B`, and `c < 2`;
* `f` is equal to zero on the boundary of the first quadrant.

Then `f` is equal to zero on the closed first quadrant. -/
theorem eq_zero_on_quadrant_I (hd : DiffContOnCl ℂ f (Ioi 0 ×ℂ Ioi 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, 0 ≤ x → f x = 0) (him : ∀ x : ℝ, 0 ≤ x → f (x * I) = 0) :
    EqOn f 0 {z | 0 ≤ z.re ∧ 0 ≤ z.im} := fun _z hz =>
  norm_le_zero_iff.1 <|
    quadrant_I hd hB (fun x hx => norm_le_zero_iff.2 <| hre x hx)
      (fun x hx => norm_le_zero_iff.2 <| him x hx) hz.1 hz.2


/-- **Phragmen-Lindelöf principle** in the first quadrant. Let `f g : ℂ → E` be functions such that

* `f` and `g` are differentiable in the open first quadrant and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * (abs z) ^ c)` on the open first
  quadrant for some `A`, `B`, and `c < 2`;
* `f` is equal to `g` on the boundary of the first quadrant.

Then `f` is equal to `g` on the closed first quadrant. -/
theorem eqOn_quadrant_I (hdf : DiffContOnCl ℂ f (Ioi 0 ×ℂ Ioi 0))
    (hBf : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hdg : DiffContOnCl ℂ g (Ioi 0 ×ℂ Ioi 0))
    (hBg : ∃ c < (2 : ℝ), ∃ B,
      g =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, 0 ≤ x → f x = g x) (him : ∀ x : ℝ, 0 ≤ x → f (x * I) = g (x * I)) :
    EqOn f g {z | 0 ≤ z.re ∧ 0 ≤ z.im} := fun _z hz =>
  sub_eq_zero.1 <|
    eq_zero_on_quadrant_I (hdf.sub hdg) (isBigO_sub_exp_rpow hBf hBg)
      (fun x hx => sub_eq_zero.2 <| hre x hx) (fun x hx => sub_eq_zero.2 <| him x hx) hz


/-- **Phragmen-Lindelöf principle** in the second quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open second quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open second quadrant
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant `C` on the boundary of the second quadrant.

Then `‖f z‖` is bounded from above by the same constant on the closed second quadrant. -/
theorem quadrant_II (hd : DiffContOnCl ℂ f (Iio 0 ×ℂ Ioi 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, x ≤ 0 → ‖f x‖ ≤ C) (him : ∀ x : ℝ, 0 ≤ x → ‖f (x * I)‖ ≤ C) (hz_re : z.re ≤ 0)
    (hz_im : 0 ≤ z.im) : ‖f z‖ ≤ C := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    hz_re : LE.le z.re 0
    hz_im : LE.le 0 z.im
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  obtain ⟨z, rfl⟩ : ∃ z', z' * I = z := ⟨z / I, div_mul_cancel₀ _ I_ne_zero⟩
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le (HMul.hMul z Complex.I).re 0
    hz_im : LE.le 0 (HMul.hMul z Complex.I).im
    ⊢ LE.le (Norm.norm (f (HMul.hMul z Complex.I))) C
  -/
  simp only [mul_I_re, mul_I_im, neg_nonpos] at hz_re hz_im
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le 0 z.im
    hz_im : LE.le 0 z.re
    ⊢ LE.le (Norm.norm (f (HMul.hMul z Complex.I))) C
  -/
  change ‖(f ∘ (· * I)) z‖ ≤ C
  have H : MapsTo (· * I) (Ioi 0 ×ℂ Ioi 0) (Iio 0 ×ℂ Ioi 0) := fun w hw ↦ by
    simpa only [mem_reProdIm, mul_I_re, mul_I_im, neg_lt_zero, mem_Iio] using hw.symm
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le 0 z.im
    hz_im : LE.le 0 z.re
    H : Set.MapsTo (fun x => HMul.hMul x Complex.I) (Complex.reProdIm (Set.Ioi 0)  …
    ⊢ LE.le (Norm.norm (Function.comp f (fun x => HMul.hMul x Complex.I) z)) C
  -/
  rcases hB with ⟨c, hc, B, hO⟩
  refine quadrant_I (hd.comp (differentiable_id.mul_const _).diffContOnCl H) ⟨c, hc, B, ?_⟩ him
    (fun x hx => ?_) hz_im hz_re
  · simpa only [Function.comp_def, map_mul, abs_I, mul_one]
      using hO.comp_tendsto ((tendsto_mul_right_cobounded I_ne_zero).inf H.tendsto)
    /-
      case intro.intro.intro.intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0))
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.im
      hz_im : LE.le 0 z.re
      H : Set.MapsTo (fun x => HMul.hMul x Complex.I) (Complex.reProdIm (Set.Ioi 0)  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (Function.comp f (fun x => HMul.hMul x Complex.I) (HMul.hMu …
    -/
  · rw [comp_apply, mul_assoc, I_mul_I, mul_neg_one, ← ofReal_neg]
    /-
      case intro.intro.intro.intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0))
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.im
      hz_im : LE.le 0 z.re
      H : Set.MapsTo (fun x => HMul.hMul x Complex.I) (Complex.reProdIm (Set.Ioi 0)  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (f ↑(Neg.neg x))) C
    -/
    exact hre _ (neg_nonpos.2 hx)
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the second quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open second quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open second quadrant
  for some `A`, `B`, and `c < 2`;
* `f` is equal to zero on the boundary of the second quadrant.

Then `f` is equal to zero on the closed second quadrant. -/
theorem eq_zero_on_quadrant_II (hd : DiffContOnCl ℂ f (Iio 0 ×ℂ Ioi 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, x ≤ 0 → f x = 0) (him : ∀ x : ℝ, 0 ≤ x → f (x * I) = 0) :
    EqOn f 0 {z | z.re ≤ 0 ∧ 0 ≤ z.im} := fun _z hz =>
  norm_le_zero_iff.1 <|
    quadrant_II hd hB (fun x hx => norm_le_zero_iff.2 <| hre x hx)
      (fun x hx => norm_le_zero_iff.2 <| him x hx) hz.1 hz.2


/-- **Phragmen-Lindelöf principle** in the second quadrant. Let `f g : ℂ → E` be functions such that

* `f` and `g` are differentiable in the open second quadrant and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * (abs z) ^ c)` on the open second
  quadrant for some `A`, `B`, and `c < 2`;
* `f` is equal to `g` on the boundary of the second quadrant.

Then `f` is equal to `g` on the closed second quadrant. -/
theorem eqOn_quadrant_II (hdf : DiffContOnCl ℂ f (Iio 0 ×ℂ Ioi 0))
    (hBf : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hdg : DiffContOnCl ℂ g (Iio 0 ×ℂ Ioi 0))
    (hBg : ∃ c < (2 : ℝ), ∃ B,
      g =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Ioi 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, x ≤ 0 → f x = g x) (him : ∀ x : ℝ, 0 ≤ x → f (x * I) = g (x * I)) :
    EqOn f g {z | z.re ≤ 0 ∧ 0 ≤ z.im} := fun _z hz =>
  sub_eq_zero.1 <| eq_zero_on_quadrant_II (hdf.sub hdg) (isBigO_sub_exp_rpow hBf hBg)
    (fun x hx => sub_eq_zero.2 <| hre x hx) (fun x hx => sub_eq_zero.2 <| him x hx) hz


/-- **Phragmen-Lindelöf principle** in the third quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open third quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp (B * (abs z) ^ c)` on the open third quadrant
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant `C` on the boundary of the third quadrant.

Then `‖f z‖` is bounded from above by the same constant on the closed third quadrant. -/
theorem quadrant_III (hd : DiffContOnCl ℂ f (Iio 0 ×ℂ Iio 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, x ≤ 0 → ‖f x‖ ≤ C) (him : ∀ x : ℝ, x ≤ 0 → ‖f (x * I)‖ ≤ C) (hz_re : z.re ≤ 0)
    (hz_im : z.im ≤ 0) : ‖f z‖ ≤ C := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    hz_re : LE.le z.re 0
    hz_im : LE.le z.im 0
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  obtain ⟨z, rfl⟩ : ∃ z', -z' = z := ⟨-z, neg_neg z⟩
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le (Neg.neg z).re 0
    hz_im : LE.le (Neg.neg z).im 0
    ⊢ LE.le (Norm.norm (f (Neg.neg z))) C
  -/
  simp only [neg_re, neg_im, neg_nonpos] at hz_re hz_im
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le 0 z.re
    hz_im : LE.le 0 z.im
    ⊢ LE.le (Norm.norm (f (Neg.neg z))) C
  -/
  change ‖(f ∘ Neg.neg) z‖ ≤ C
  have H : MapsTo Neg.neg (Ioi 0 ×ℂ Ioi 0) (Iio 0 ×ℂ Iio 0) := by
    intro w hw
    simpa only [mem_reProdIm, neg_re, neg_im, neg_lt_zero, mem_Iio] using hw
  refine
    quadrant_I (hd.comp differentiable_neg.diffContOnCl H) ?_ (fun x hx => ?_) (fun x hx => ?_)
      hz_re hz_im
    /-
      case intro.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.re
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) (Complex.reP …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
  · rcases hB with ⟨c, hc, B, hO⟩
    /-
      case intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.re
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) (Complex.reP …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
    refine ⟨c, hc, B, ?_⟩
    simpa only [Function.comp_def, Complex.abs.map_neg]
      using hO.comp_tendsto (tendsto_neg_cobounded.inf H.tendsto)
    /-
      case intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.re
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (Function.comp f Neg.neg ↑x)) C
    -/
  · rw [comp_apply, ← ofReal_neg]
    /-
      case intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.re
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (f ↑(Neg.neg x))) C
    -/
    exact hre (-x) (neg_nonpos.2 hx)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.re
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (Function.comp f Neg.neg (HMul.hMul (↑x) Complex.I))) C
    -/
  · rw [comp_apply, ← neg_mul, ← ofReal_neg]
    /-
      case intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Iio 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le 0 z.re
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Ioi 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (f (HMul.hMul (↑(Neg.neg x)) Complex.I))) C
    -/
    exact him (-x) (neg_nonpos.2 hx)
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the third quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open third quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open third quadrant
  for some `A`, `B`, and `c < 2`;
* `f` is equal to zero on the boundary of the third quadrant.

Then `f` is equal to zero on the closed third quadrant. -/
theorem eq_zero_on_quadrant_III (hd : DiffContOnCl ℂ f (Iio 0 ×ℂ Iio 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, x ≤ 0 → f x = 0) (him : ∀ x : ℝ, x ≤ 0 → f (x * I) = 0) :
    EqOn f 0 {z | z.re ≤ 0 ∧ z.im ≤ 0} := fun _z hz =>
  norm_le_zero_iff.1 <| quadrant_III hd hB (fun x hx => norm_le_zero_iff.2 <| hre x hx)
    (fun x hx => norm_le_zero_iff.2 <| him x hx) hz.1 hz.2


/-- **Phragmen-Lindelöf principle** in the third quadrant. Let `f g : ℂ → E` be functions such that

* `f` and `g` are differentiable in the open third quadrant and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * (abs z) ^ c)` on the open third
  quadrant for some `A`, `B`, and `c < 2`;
* `f` is equal to `g` on the boundary of the third quadrant.

Then `f` is equal to `g` on the closed third quadrant. -/
theorem eqOn_quadrant_III (hdf : DiffContOnCl ℂ f (Iio 0 ×ℂ Iio 0))
    (hBf : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hdg : DiffContOnCl ℂ g (Iio 0 ×ℂ Iio 0))
    (hBg : ∃ c < (2 : ℝ), ∃ B,
      g =O[cobounded ℂ ⊓ 𝓟 (Iio 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, x ≤ 0 → f x = g x) (him : ∀ x : ℝ, x ≤ 0 → f (x * I) = g (x * I)) :
    EqOn f g {z | z.re ≤ 0 ∧ z.im ≤ 0} := fun _z hz =>
  sub_eq_zero.1 <| eq_zero_on_quadrant_III (hdf.sub hdg) (isBigO_sub_exp_rpow hBf hBg)
    (fun x hx => sub_eq_zero.2 <| hre x hx) (fun x hx => sub_eq_zero.2 <| him x hx) hz


/-- **Phragmen-Lindelöf principle** in the fourth quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open fourth quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open fourth quadrant
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant `C` on the boundary of the fourth quadrant.

Then `‖f z‖` is bounded from above by the same constant on the closed fourth quadrant. -/
theorem quadrant_IV (hd : DiffContOnCl ℂ f (Ioi 0 ×ℂ Iio 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, 0 ≤ x → ‖f x‖ ≤ C) (him : ∀ x : ℝ, x ≤ 0 → ‖f (x * I)‖ ≤ C) (hz_re : 0 ≤ z.re)
    (hz_im : z.im ≤ 0) : ‖f z‖ ≤ C := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    hz_re : LE.le 0 z.re
    hz_im : LE.le z.im 0
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  obtain ⟨z, rfl⟩ : ∃ z', -z' = z := ⟨-z, neg_neg z⟩
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le 0 (Neg.neg z).re
    hz_im : LE.le (Neg.neg z).im 0
    ⊢ LE.le (Norm.norm (f (Neg.neg z))) C
  -/
  simp only [neg_re, neg_im, neg_nonpos, neg_nonneg] at hz_re hz_im
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
    hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
    hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
    him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
    z : Complex
    hz_re : LE.le z.re 0
    hz_im : LE.le 0 z.im
    ⊢ LE.le (Norm.norm (f (Neg.neg z))) C
  -/
  change ‖(f ∘ Neg.neg) z‖ ≤ C
  have H : MapsTo Neg.neg (Iio 0 ×ℂ Ioi 0) (Ioi 0 ×ℂ Iio 0) := fun w hw ↦ by
    simpa only [mem_reProdIm, neg_re, neg_im, neg_lt_zero, neg_pos, mem_Ioi, mem_Iio] using hw
  refine quadrant_II
    (hd.comp differentiable_neg.diffContOnCl H) ?_ (fun x hx => ?_) (fun x hx => ?_) hz_re hz_im
    /-
      case intro.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le z.re 0
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0)) (Complex.reP …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
  · rcases hB with ⟨c, hc, B, hO⟩
    /-
      case intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le z.re 0
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0)) (Complex.reP …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
    refine ⟨c, hc, B, ?_⟩
    simpa only [Function.comp_def, Complex.abs.map_neg]
      using hO.comp_tendsto (tendsto_neg_cobounded.inf H.tendsto)
    /-
      case intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le z.re 0
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le x 0
      ⊢ LE.le (Norm.norm (Function.comp f Neg.neg ↑x)) C
    -/
  · rw [comp_apply, ← ofReal_neg]
    /-
      case intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le z.re 0
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le x 0
      ⊢ LE.le (Norm.norm (f ↑(Neg.neg x))) C
    -/
    exact hre (-x) (neg_nonneg.2 hx)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le z.re 0
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (Function.comp f Neg.neg (HMul.hMul (↑x) Complex.I))) C
    -/
  · rw [comp_apply, ← neg_mul, ← ofReal_neg]
    /-
      case intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (Complex.reProdIm (Set.Ioi 0) (Set.Iio 0))
      hB : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min. …
      hre : ∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C
      him : ∀ (x : Real), LE.le x 0 → LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I) …
      z : Complex
      hz_re : LE.le z.re 0
      hz_im : LE.le 0 z.im
      H : Set.MapsTo Neg.neg (Complex.reProdIm (Set.Iio 0) (Set.Ioi 0)) (Complex.reP …
      x : Real
      hx : LE.le 0 x
      ⊢ LE.le (Norm.norm (f (HMul.hMul (↑(Neg.neg x)) Complex.I))) C
    -/
    exact him (-x) (neg_nonpos.2 hx)
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the fourth quadrant. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open fourth quadrant and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open fourth quadrant
  for some `A`, `B`, and `c < 2`;
* `f` is equal to zero on the boundary of the fourth quadrant.

Then `f` is equal to zero on the closed fourth quadrant. -/
theorem eq_zero_on_quadrant_IV (hd : DiffContOnCl ℂ f (Ioi 0 ×ℂ Iio 0))
    (hB : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, 0 ≤ x → f x = 0) (him : ∀ x : ℝ, x ≤ 0 → f (x * I) = 0) :
    EqOn f 0 {z | 0 ≤ z.re ∧ z.im ≤ 0} := fun _z hz =>
  norm_le_zero_iff.1 <|
    quadrant_IV hd hB (fun x hx => norm_le_zero_iff.2 <| hre x hx)
      (fun x hx => norm_le_zero_iff.2 <| him x hx) hz.1 hz.2


/-- **Phragmen-Lindelöf principle** in the fourth quadrant. Let `f g : ℂ → E` be functions such that

* `f` and `g` are differentiable in the open fourth quadrant and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * (abs z) ^ c)` on the open fourth
  quadrant for some `A`, `B`, and `c < 2`;
* `f` is equal to `g` on the boundary of the fourth quadrant.

Then `f` is equal to `g` on the closed fourth quadrant. -/
theorem eqOn_quadrant_IV (hdf : DiffContOnCl ℂ f (Ioi 0 ×ℂ Iio 0))
    (hBf : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hdg : DiffContOnCl ℂ g (Ioi 0 ×ℂ Iio 0))
    (hBg : ∃ c < (2 : ℝ), ∃ B,
      g =O[cobounded ℂ ⊓ 𝓟 (Ioi 0 ×ℂ Iio 0)] fun z => expR (B * abs z ^ c))
    (hre : ∀ x : ℝ, 0 ≤ x → f x = g x) (him : ∀ x : ℝ, x ≤ 0 → f (x * I) = g (x * I)) :
    EqOn f g {z | 0 ≤ z.re ∧ z.im ≤ 0} := fun _z hz =>
  sub_eq_zero.1 <| eq_zero_on_quadrant_IV (hdf.sub hdg) (isBigO_sub_exp_rpow hBf hBg)
    (fun x hx => sub_eq_zero.2 <| hre x hx) (fun x hx => sub_eq_zero.2 <| him x hx) hz


/-- **Phragmen-Lindelöf principle** in the right half-plane. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open right half-plane and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open right half-plane
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant `C` on the imaginary axis;
* `f x → 0` as `x : ℝ` tends to infinity.

Then `‖f z‖` is bounded from above by the same constant on the closed right half-plane.
See also `PhragmenLindelof.right_half_plane_of_bounded_on_real` for a stronger version. -/
theorem right_half_plane_of_tendsto_zero_on_real (hd : DiffContOnCl ℂ f {z | 0 < z.re})
    (hexp : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 {z | 0 < z.re}] fun z => expR (B * abs z ^ c))
    (hre : Tendsto (fun x : ℝ => f x) atTop (𝓝 0)) (him : ∀ x : ℝ, ‖f (x * I)‖ ≤ C)
    (hz : 0 ≤ z.re) : ‖f z‖ ≤ C := by
  /- We are going to apply the Phragmen-Lindelöf principle in the first and fourth quadrants.
    The lemmas immediately imply that for any upper estimate `C'` on `‖f x‖`, `x : ℝ`, `0 ≤ x`,
    the number `max C C'` is an upper estimate on `f` in the whole right half-plane. -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Filter.Tendsto (fun x => f ↑x) Filter.atTop (nhds 0)
    him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    hz : LE.le 0 z.re
    ⊢ LE.le (Norm.norm (f z)) C
  -/
  revert z
  have hle : ∀ C', (∀ x : ℝ, 0 ≤ x → ‖f x‖ ≤ C') →
      ∀ z : ℂ, 0 ≤ z.re → ‖f z‖ ≤ max C C' := fun C' hC' z hz ↦ by
    rcases hexp with ⟨c, hc, B, hO⟩
    rcases le_total z.im 0 with h | h
    · refine quadrant_IV (hd.mono fun _ => And.left) ⟨c, hc, B, ?_⟩
          (fun x hx => (hC' x hx).trans <| le_max_right _ _)
          (fun x _ => (him x).trans (le_max_left _ _)) hz h
      exact hO.mono (inf_le_inf_left _ <| principal_mono.2 fun _ => And.left)
    · refine quadrant_I (hd.mono fun _ => And.left) ⟨c, hc, B, ?_⟩
          (fun x hx => (hC' x hx).trans <| le_max_right _ _)
          (fun x _ => (him x).trans (le_max_left _ _)) hz h
      exact hO.mono (inf_le_inf_left _ <| principal_mono.2 fun _ => And.left)
  -- Since `f` is continuous on `Ici 0` and `‖f x‖` tends to zero as `x → ∞`,
  -- the norm `‖f x‖` takes its maximum value at some `x₀ : ℝ`.
  obtain ⟨x₀, hx₀, hmax⟩ : ∃ x : ℝ, 0 ≤ x ∧ ∀ y : ℝ, 0 ≤ y → ‖f y‖ ≤ ‖f x‖ := by
    have hfc : ContinuousOn (fun x : ℝ => f x) (Ici 0) := by
      refine hd.continuousOn.comp continuous_ofReal.continuousOn fun x hx => ?_
      rwa [closure_setOf_lt_re]
    by_cases h₀ : ∀ x : ℝ, 0 ≤ x → f x = 0
    · refine ⟨0, le_rfl, fun y hy => ?_⟩; rw [h₀ y hy, h₀ 0 le_rfl]
    push_neg at h₀
    rcases h₀ with ⟨x₀, hx₀, hne⟩
    have hlt : ‖(0 : E)‖ < ‖f x₀‖ := by rwa [norm_zero, norm_pos_iff]
    suffices ∀ᶠ x : ℝ in cocompact ℝ ⊓ 𝓟 (Ici 0), ‖f x‖ ≤ ‖f x₀‖ by
      simpa only [exists_prop] using hfc.norm.exists_isMaxOn' isClosed_Ici hx₀ this
    rw [cocompact_eq_atBot_atTop, inf_sup_right, (disjoint_atBot_principal_Ici (0 : ℝ)).eq_bot,
      bot_sup_eq]
    exact (hre.norm.eventually <| ge_mem_nhds hlt).filter_mono inf_le_left
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Filter.Tendsto (fun x => f ↑x) Filter.atTop (nhds 0)
    him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    hle : ∀ (C' : Real), (∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C') → …
    x₀ : Real
    hx₀ : LE.le 0 x₀
    hmax : ∀ (y : Real), LE.le 0 y → LE.le (Norm.norm (f ↑y)) (Norm.norm (f ↑x₀))
    ⊢ ∀ {z : Complex}, LE.le 0 z.re → LE.le (Norm.norm (f z)) C
  -/
  rcases le_or_lt ‖f x₀‖ C with h | h
  ·-- If `‖f x₀‖ ≤ C`, then `hle` implies the required estimate
    /-
      case intro.intro.inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.Tendsto (fun x => f ↑x) Filter.atTop (nhds 0)
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hle : ∀ (C' : Real), (∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C') → …
      x₀ : Real
      hx₀ : LE.le 0 x₀
      hmax : ∀ (y : Real), LE.le 0 y → LE.le (Norm.norm (f ↑y)) (Norm.norm (f ↑x₀))
      h : LE.le (Norm.norm (f ↑x₀)) C
      ⊢ ∀ {z : Complex}, LE.le 0 z.re → LE.le (Norm.norm (f z)) C
    -/
    simpa only [max_eq_left h] using hle _ hmax
    /-
      🎉 no goals
    -/
  · -- Otherwise, `‖f z‖ ≤ ‖f x₀‖` for all `z` in the right half-plane due to `hle`.
    replace hmax : IsMaxOn (norm ∘ f) {z | 0 < z.re} x₀ := by
      rintro z (hz : 0 < z.re)
      simpa [max_eq_right h.le] using hle _ hmax _ hz.le
    -- Due to the maximum modulus principle applied to the closed ball of radius `x₀.re`,
    -- `‖f 0‖ = ‖f x₀‖`.
    have : ‖f 0‖ = ‖f x₀‖ := by
      apply norm_eq_norm_of_isMaxOn_of_ball_subset hd hmax
      -- move to a lemma?
      intro z hz
      rw [mem_ball, dist_zero_left, dist_eq, norm_eq_abs, Complex.abs_of_nonneg hx₀] at hz
      rw [mem_setOf_eq]
      contrapose! hz
      calc
        x₀ ≤ x₀ - z.re := (le_sub_self_iff _).2 hz
        _ ≤ |x₀ - z.re| := le_abs_self _
        _ = |(z - x₀).re| := by rw [sub_re, ofReal_re, _root_.abs_sub_comm]
        _ ≤ abs (z - x₀) := abs_re_le_abs _
    -- Thus we have `C < ‖f x₀‖ = ‖f 0‖ ≤ C`. Contradiction completes the proof.
    /-
      case intro.intro.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.Tendsto (fun x => f ↑x) Filter.atTop (nhds 0)
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hle : ∀ (C' : Real), (∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C') → …
      x₀ : Real
      hx₀ : LE.le 0 x₀
      h : LT.lt C (Norm.norm (f ↑x₀))
      hmax : IsMaxOn (Function.comp Norm.norm f) (setOf fun z => LT.lt 0 z.re) ↑x₀
      this : Eq (Norm.norm (f 0)) (Norm.norm (f ↑x₀))
      ⊢ ∀ {z : Complex}, LE.le 0 z.re → LE.le (Norm.norm (f z)) C
    -/
    refine (h.not_le <| this ▸ ?_).elim
    /-
      case intro.intro.inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.Tendsto (fun x => f ↑x) Filter.atTop (nhds 0)
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hle : ∀ (C' : Real), (∀ (x : Real), LE.le 0 x → LE.le (Norm.norm (f ↑x)) C') → …
      x₀ : Real
      hx₀ : LE.le 0 x₀
      h : LT.lt C (Norm.norm (f ↑x₀))
      hmax : IsMaxOn (Function.comp Norm.norm f) (setOf fun z => LT.lt 0 z.re) ↑x₀
      this : Eq (Norm.norm (f 0)) (Norm.norm (f ↑x₀))
      ⊢ LE.le (Norm.norm (f 0)) C
    -/
    simpa using him 0
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the right half-plane. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open right half-plane and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open right half-plane
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant `C` on the imaginary axis;
* `‖f x‖` is bounded from above by a constant for large real values of `x`.

Then `‖f z‖` is bounded from above by `C` on the closed right half-plane.
See also `PhragmenLindelof.right_half_plane_of_tendsto_zero_on_real` for a weaker version. -/
theorem right_half_plane_of_bounded_on_real (hd : DiffContOnCl ℂ f {z | 0 < z.re})
    (hexp : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 {z | 0 < z.re}] fun z => expR (B * abs z ^ c))
    (hre : IsBoundedUnder (· ≤ ·) atTop fun x : ℝ => ‖f x‖) (him : ∀ x : ℝ, ‖f (x * I)‖ ≤ C)
    (hz : 0 ≤ z.re) : ‖f z‖ ≤ C := by
  -- For each `ε < 0`, the function `fun z ↦ exp (ε * z) • f z` satisfies assumptions of
  -- `right_half_plane_of_tendsto_zero_on_real`, hence `‖exp (ε * z) • f z‖ ≤ C` for all `ε < 0`.
  -- Taking the limit as `ε → 0`, we obtain the required inequality.
  suffices ∀ᶠ ε : ℝ in 𝓝[<] 0, ‖exp (ε * z) • f z‖ ≤ C by
    refine le_of_tendsto (Tendsto.mono_left ?_ nhdsWithin_le_nhds) this
    apply ((continuous_ofReal.mul continuous_const).cexp.smul continuous_const).norm.tendsto'
    simp
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
    him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    hz : LE.le 0 z.re
    ⊢ Filter.Eventually (fun ε => LE.le (Norm.norm (HSMul.hSMul (Complex.exp (HMul …
  -/
  filter_upwards [self_mem_nhdsWithin] with ε ε₀; change ε < 0 at ε₀
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
    him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    hz : LE.le 0 z.re
    ε : Real
    ε₀ : LT.lt ε 0
    ⊢ LE.le (Norm.norm (HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z))) C
  -/
  set g : ℂ → E := fun z => exp (ε * z) • f z; change ‖g z‖ ≤ C
  replace hd : DiffContOnCl ℂ g {z : ℂ | 0 < z.re} :=
    (differentiable_id.const_mul _).cexp.diffContOnCl.smul hd
  have hgn : ∀ z, ‖g z‖ = expR (ε * z.re) * ‖f z‖ := fun z ↦ by
    rw [norm_smul, norm_eq_abs, abs_exp, re_ofReal_mul]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    C : Real
    f : Complex → E
    z : Complex
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
    him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    hz : LE.le 0 z.re
    ε : Real
    ε₀ : LT.lt ε 0
    g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
    hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
    hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
    ⊢ LE.le (Norm.norm (g z)) C
  -/
  refine right_half_plane_of_tendsto_zero_on_real hd ?_ ?_ (fun y => ?_) hz
    /-
      case h.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z : Complex
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz : LE.le 0 z.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
  · rcases hexp with ⟨c, hc, B, hO⟩
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z : Complex
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz : LE.le 0 z.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
    refine ⟨c, hc, B, (IsBigO.of_bound 1 ?_).trans hO⟩
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z : Complex
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz : LE.le 0 z.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (HMul.hMul 1 (Norm.norm  …
    -/
    refine eventually_inf_principal.2 <| Eventually.of_forall fun z hz => ?_
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z✝ : Complex
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz✝ : LE.le 0 z✝.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      z : Complex
      hz : Membership.mem (setOf fun z => LT.lt 0 z.re) z
      ⊢ LE.le (Norm.norm (g z)) (HMul.hMul 1 (Norm.norm (f z)))
    -/
    rw [hgn, one_mul]
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z✝ : Complex
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz✝ : LE.le 0 z✝.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      z : Complex
      hz : Membership.mem (setOf fun z => LT.lt 0 z.re) z
      ⊢ LE.le (HMul.hMul (Real.exp (HMul.hMul ε z.re)) (Norm.norm (f z))) (Norm.norm …
    -/
    refine mul_le_of_le_one_left (norm_nonneg _) (Real.exp_le_one_iff.2 ?_)
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z✝ : Complex
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz✝ : LE.le 0 z✝.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      z : Complex
      hz : Membership.mem (setOf fun z => LT.lt 0 z.re) z
      ⊢ LE.le (HMul.hMul ε z.re) 0
    -/
    exact mul_nonpos_of_nonpos_of_nonneg ε₀.le (le_of_lt hz)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z : Complex
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz : LE.le 0 z.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      ⊢ Filter.Tendsto (fun x => g ↑x) Filter.atTop (nhds 0)
    -/
  · simp_rw [g, ← ofReal_mul, ← ofReal_exp, coe_smul]
    have h₀ : Tendsto (fun x : ℝ => expR (ε * x)) atTop (𝓝 0) :=
      Real.tendsto_exp_atBot.comp (tendsto_const_nhds.neg_mul_atTop ε₀ tendsto_id)
    /-
      case h.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z : Complex
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz : LE.le 0 z.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      h₀ : Filter.Tendsto (fun x => Real.exp (HMul.hMul ε x)) Filter.atTop (nhds 0)
      ⊢ Filter.Tendsto (fun x => HSMul.hSMul (Real.exp (HMul.hMul ε x)) (f ↑x)) Filt …
    -/
    exact h₀.zero_smul_isBoundedUnder_le hre
    /-
      🎉 no goals
    -/
  · rw [hgn, re_ofReal_mul, I_re, mul_zero, mul_zero, Real.exp_zero,
      one_mul]
    /-
      case h.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      C : Real
      f : Complex → E
      z : Complex
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun x => N …
      him : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      hz : LE.le 0 z.re
      ε : Real
      ε₀ : LT.lt ε 0
      g : Complex → E := fun z => HSMul.hSMul (Complex.exp (HMul.hMul (↑ε) z)) (f z)
      hd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hgn : ∀ (z : Complex), Eq (Norm.norm (g z)) (HMul.hMul (Real.exp (HMul.hMul ε  …
      y : Real
      ⊢ LE.le (Norm.norm (f (HMul.hMul (↑y) Complex.I))) C
    -/
    exact him y
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the right half-plane. Let `f : ℂ → E` be a function such that

* `f` is differentiable in the open right half-plane and is continuous on its closure;
* `‖f z‖` is bounded from above by `A * exp(B * (abs z) ^ c)` on the open right half-plane
  for some `c < 2`;
* `‖f z‖` is bounded from above by a constant on the imaginary axis;
* `f x`, `x : ℝ`, tends to zero superexponentially fast as `x → ∞`:
  for any natural `n`, `exp (n * x) * ‖f x‖` tends to zero as `x → ∞`.

Then `f` is equal to zero on the closed right half-plane. -/
theorem eq_zero_on_right_half_plane_of_superexponential_decay (hd : DiffContOnCl ℂ f {z | 0 < z.re})
    (hexp : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 {z | 0 < z.re}] fun z => expR (B * abs z ^ c))
    (hre : SuperpolynomialDecay atTop expR fun x => ‖f x‖) (him : ∃ C, ∀ x : ℝ, ‖f (x * I)‖ ≤ C) :
    EqOn f 0 {z : ℂ | 0 ≤ z.re} := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
    him : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Comple …
    ⊢ Set.EqOn f 0 (setOf fun z => LE.le 0 z.re)
  -/
  rcases him with ⟨C, hC⟩
  -- Due to continuity, it suffices to prove the equality on the open right half-plane.
  suffices ∀ z : ℂ, 0 < z.re → f z = 0 by
    simpa only [closure_setOf_lt_re] using
      EqOn.of_subset_closure this hd.continuousOn continuousOn_const subset_closure Subset.rfl
  -- Consider $g_n(z)=e^{nz}f(z)$.
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
    C : Real
    hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    ⊢ ∀ (z : Complex), LT.lt 0 z.re → Eq (f z) 0
  -/
  set g : ℕ → ℂ → E := fun (n : ℕ) (z : ℂ) => exp z ^ n • f z
  have hg : ∀ n z, ‖g n z‖ = expR z.re ^ n * ‖f z‖ := fun n z ↦ by
    simp only [g, norm_smul, norm_eq_abs, Complex.abs_pow, abs_exp]
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
    C : Real
    hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
    hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
    ⊢ ∀ (z : Complex), LT.lt 0 z.re → Eq (f z) 0
  -/
  intro z hz
  -- Since `e^{nz} → ∞` as `n → ∞`, it suffices to show that each `g_n` is bounded from above by `C`
  suffices H : ∀ n : ℕ, ‖g n z‖ ≤ C by
    contrapose! H
    simp only [hg]
    exact (((tendsto_pow_atTop_atTop_of_one_lt (Real.one_lt_exp_iff.2 hz)).atTop_mul
      (norm_pos_iff.2 H) tendsto_const_nhds).eventually (eventually_gt_atTop C)).exists
  /-
    case intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
    hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
    C : Real
    hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
    g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
    hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
    z : Complex
    hz : LT.lt 0 z.re
    ⊢ ∀ (n : Nat), LE.le (Norm.norm (g n z)) C
  -/
  intro n
  -- This estimate follows from the Phragmen-Lindelöf principle in the right half-plane.
  refine right_half_plane_of_tendsto_zero_on_real ((differentiable_exp.pow n).diffContOnCl.smul hd)
    ?_ ?_ (fun y => ?_) hz.le
    /-
      case intro.refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
  · rcases hexp with ⟨c, hc, B, hO⟩
    /-
      case intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
    refine ⟨max c 1, max_lt hc one_lt_two, n + max B 0, .of_norm_left ?_⟩
    /-
      case intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.principal  …
    -/
    simp only [hg]
    /-
      case intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.principal  …
    -/
    refine ((isBigO_refl (fun z : ℂ => expR z.re ^ n) _).mul hO.norm_left).trans (.of_bound 1 ?_)
    /-
      case intro.refine_1.intro.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HMul.hMul (HPow.hPow (Real.exp …
    -/
    filter_upwards [(eventually_cobounded_le_norm 1).filter_mono inf_le_left] with z hz
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z✝ : Complex
      hz✝ : LT.lt 0 z✝.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      z : Complex
      hz : LE.le 1 (Norm.norm z)
      ⊢ LE.le (Norm.norm (HMul.hMul (HPow.hPow (Real.exp z.re) n) (Real.exp (HMul.hM …
    -/
    simp only [← Real.exp_nat_mul, ← Real.exp_add, Real.norm_eq_abs, Real.abs_exp, add_mul, one_mul]
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z✝ : Complex
      hz✝ : LT.lt 0 z✝.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      z : Complex
      hz : LE.le 1 (Norm.norm z)
      ⊢ LE.le (Real.exp (HAdd.hAdd (HMul.hMul (↑n) z.re) (HMul.hMul B (HPow.hPow (Co …
    -/
    gcongr
    · calc
        z.re ≤ abs z := re_le_abs _
        _ = abs z ^ (1 : ℝ) := (Real.rpow_one _).symm
        _ ≤ abs z ^ max c 1 := Real.rpow_le_rpow_of_exponent_le hz (le_max_right _ _)
    /-
      case h.h.h₂.h₁
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z✝ : Complex
      hz✝ : LT.lt 0 z✝.re
      n : Nat
      c : Real
      hc : LT.lt c 2
      B : Real
      hO : Asymptotics.IsBigO (Min.min (Bornology.cobounded Complex) (Filter.princip …
      z : Complex
      hz : LE.le 1 (Norm.norm z)
      ⊢ LE.le B (Max.max B 0)
    -/
    exacts [le_max_left _ _, hz, le_max_left _ _]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      ⊢ Filter.Tendsto (fun x => g n ↑x) Filter.atTop (nhds 0)
    -/
  · rw [tendsto_zero_iff_norm_tendsto_zero]; simp only [hg]
    /-
      case intro.refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Real.exp (↑x).re) n) (Norm.no …
    -/
    exact hre n
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      y : Real
      ⊢ LE.le (Norm.norm (g n (HMul.hMul (↑y) Complex.I))) C
    -/
  · rw [hg, re_ofReal_mul, I_re, mul_zero, Real.exp_zero, one_pow, one_mul]
    /-
      case intro.refine_3
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : Complex → E
      hd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Mi …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      C : Real
      hC : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) C
      g : Nat → Complex → E := fun n z => HSMul.hSMul (HPow.hPow (Complex.exp z) n)  …
      hg : ∀ (n : Nat) (z : Complex), Eq (Norm.norm (g n z)) (HMul.hMul (HPow.hPow ( …
      z : Complex
      hz : LT.lt 0 z.re
      n : Nat
      y : Real
      ⊢ LE.le (Norm.norm (f (HMul.hMul (↑y) Complex.I))) C
    -/
    exact hC y
    /-
      🎉 no goals
    -/


/-- **Phragmen-Lindelöf principle** in the right half-plane. Let `f g : ℂ → E` be functions such
that

* `f` and `g` are differentiable in the open right half-plane and are continuous on its closure;
* `‖f z‖` and `‖g z‖` are bounded from above by `A * exp(B * (abs z) ^ c)` on the open right
  half-plane for some `c < 2`;
* `‖f z‖` and `‖g z‖` are bounded from above by constants on the imaginary axis;
* `f x - g x`, `x : ℝ`, tends to zero superexponentially fast as `x → ∞`:
  for any natural `n`, `exp (n * x) * ‖f x - g x‖` tends to zero as `x → ∞`.

Then `f` is equal to `g` on the closed right half-plane. -/
theorem eqOn_right_half_plane_of_superexponential_decay {g : ℂ → E}
    (hfd : DiffContOnCl ℂ f {z | 0 < z.re}) (hgd : DiffContOnCl ℂ g {z | 0 < z.re})
    (hfexp : ∃ c < (2 : ℝ), ∃ B,
      f =O[cobounded ℂ ⊓ 𝓟 {z | 0 < z.re}] fun z => expR (B * abs z ^ c))
    (hgexp : ∃ c < (2 : ℝ), ∃ B,
      g =O[cobounded ℂ ⊓ 𝓟 {z | 0 < z.re}] fun z => expR (B * abs z ^ c))
    (hre : SuperpolynomialDecay atTop expR fun x => ‖f x - g x‖)
    (hfim : ∃ C, ∀ x : ℝ, ‖f (x * I)‖ ≤ C) (hgim : ∃ C, ∀ x : ℝ, ‖g (x * I)‖ ≤ C) :
    EqOn f g {z : ℂ | 0 ≤ z.re} := by
  suffices EqOn (f - g) 0 {z : ℂ | 0 ≤ z.re} by
    simpa only [EqOn, Pi.sub_apply, Pi.zero_apply, sub_eq_zero] using this
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : Complex → E
    hfd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
    hgd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
    hfexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
    hgexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
    hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
    hfim : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Compl …
    hgim : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (g (HMul.hMul (↑x) Compl …
    ⊢ Set.EqOn (HSub.hSub f g) 0 (setOf fun z => LE.le 0 z.re)
  -/
  refine eq_zero_on_right_half_plane_of_superexponential_decay (hfd.sub hgd) ?_ hre ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f g : Complex → E
      hfd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hgd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hfexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
      hgexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      hfim : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Compl …
      hgim : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (g (HMul.hMul (↑x) Compl …
      ⊢ Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (Min.min …
    -/
  · exact isBigO_sub_exp_rpow hfexp hgexp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f g : Complex → E
      hfd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hgd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hfexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
      hgexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      hfim : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Compl …
      hgim : Exists fun C => ∀ (x : Real), LE.le (Norm.norm (g (HMul.hMul (↑x) Compl …
      ⊢ Exists fun C => ∀ (x : Real), LE.le (Norm.norm (HSub.hSub f g (HMul.hMul (↑x …
    -/
  · rcases hfim with ⟨Cf, hCf⟩; rcases hgim with ⟨Cg, hCg⟩
    /-
      case refine_2.intro.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f g : Complex → E
      hfd : DiffContOnCl Complex f (setOf fun z => LT.lt 0 z.re)
      hgd : DiffContOnCl Complex g (setOf fun z => LT.lt 0 z.re)
      hfexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
      hgexp : Exists fun c => And (LT.lt c 2) (Exists fun B => Asymptotics.IsBigO (M …
      hre : Asymptotics.SuperpolynomialDecay Filter.atTop Real.exp fun x => Norm.nor …
      Cf : Real
      hCf : ∀ (x : Real), LE.le (Norm.norm (f (HMul.hMul (↑x) Complex.I))) Cf
      Cg : Real
      hCg : ∀ (x : Real), LE.le (Norm.norm (g (HMul.hMul (↑x) Complex.I))) Cg
      ⊢ Exists fun C => ∀ (x : Real), LE.le (Norm.norm (HSub.hSub f g (HMul.hMul (↑x …
    -/
    exact ⟨Cf + Cg, fun x => norm_sub_le_of_le (hCf x) (hCg x)⟩
    /-
      🎉 no goals
    -/


