theorem eventually_le_limsup [CountableInterFilter f] (u : α → ℝ≥0∞) :
    ∀ᶠ y in f, u y ≤ f.limsup u :=
  /-
    α : Type u_1
    f : Filter α
    inst✝ : CountableInterFilter f
    u : α → ENNReal
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) f u
  -/
  _root_.eventually_le_limsup
  /-
    🎉 no goals
  -/


theorem limsup_eq_zero_iff [CountableInterFilter f] {u : α → ℝ≥0∞} :
    f.limsup u = 0 ↔ u =ᶠ[f] 0 :=
  limsup_eq_bot


theorem limsup_const_mul_of_ne_top {u : α → ℝ≥0∞} {a : ℝ≥0∞} (ha_top : a ≠ ⊤) :
    (f.limsup fun x : α => a * u x) = a * f.limsup u := by
  /-
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    ha_top : Ne a Top.top
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  by_cases ha_zero : a = 0
    /-
      case pos
      α : Type u_1
      f : Filter α
      u : α → ENNReal
      a : ENNReal
      ha_top : Ne a Top.top
      ha_zero : Eq a 0
      ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
    -/
  · simp_rw [ha_zero, zero_mul, ← ENNReal.bot_eq_zero]
    /-
      case pos
      α : Type u_1
      f : Filter α
      u : α → ENNReal
      a : ENNReal
      ha_top : Ne a Top.top
      ha_zero : Eq a 0
      ⊢ Eq (Filter.limsup (fun x => Bot.bot) f) Bot.bot
    -/
    exact limsup_const_bot
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    ha_top : Ne a Top.top
    ha_zero : Not (Eq a 0)
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  let g := fun x : ℝ≥0∞ => a * x
  have hg_bij : Function.Bijective g :=
    Function.bijective_iff_has_inverse.mpr
      ⟨fun x => a⁻¹ * x,
        ⟨fun x => by simp [g, ← mul_assoc, ENNReal.inv_mul_cancel ha_zero ha_top], fun x => by
          simp [g, ← mul_assoc, ENNReal.mul_inv_cancel ha_zero ha_top]⟩⟩
  have hg_mono : StrictMono g :=
    Monotone.strictMono_of_injective (fun _ _ _ => by rwa [mul_le_mul_left ha_zero ha_top]) hg_bij.1
  /-
    case neg
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    ha_top : Ne a Top.top
    ha_zero : Not (Eq a 0)
    g : ENNReal → ENNReal := fun x => HMul.hMul a x
    hg_bij : Function.Bijective g
    hg_mono : StrictMono g
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  let g_iso := StrictMono.orderIsoOfSurjective g hg_mono hg_bij.2
  /-
    case neg
    α : Type u_1
    f : Filter α
    u : α → ENNReal
    a : ENNReal
    ha_top : Ne a Top.top
    ha_zero : Not (Eq a 0)
    g : ENNReal → ENNReal := fun x => HMul.hMul a x
    hg_bij : Function.Bijective g
    hg_mono : StrictMono g
    g_iso : OrderIso ENNReal ENNReal := StrictMono.orderIsoOfSurjective g hg_mono ⋯
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  exact (OrderIso.limsup_apply g_iso).symm
  /-
    🎉 no goals
  -/


theorem limsup_const_mul [CountableInterFilter f] {u : α → ℝ≥0∞} {a : ℝ≥0∞} :
    f.limsup (a * u ·) = a * f.limsup u := by
  /-
    α : Type u_1
    f : Filter α
    inst✝ : CountableInterFilter f
    u : α → ENNReal
    a : ENNReal
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  by_cases ha_top : a ≠ ⊤
    /-
      case pos
      α : Type u_1
      f : Filter α
      inst✝ : CountableInterFilter f
      u : α → ENNReal
      a : ENNReal
      ha_top : Ne a Top.top
      ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
    -/
  · exact limsup_const_mul_of_ne_top ha_top
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    f : Filter α
    inst✝ : CountableInterFilter f
    u : α → ENNReal
    a : ENNReal
    ha_top : Not (Ne a Top.top)
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  push_neg at ha_top
  /-
    case neg
    α : Type u_1
    f : Filter α
    inst✝ : CountableInterFilter f
    u : α → ENNReal
    a : ENNReal
    ha_top : Eq a Top.top
    ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
  -/
  by_cases hu : u =ᶠ[f] 0
    /-
      case pos
      α : Type u_1
      f : Filter α
      inst✝ : CountableInterFilter f
      u : α → ENNReal
      a : ENNReal
      ha_top : Eq a Top.top
      hu : f.EventuallyEq u 0
      ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
    -/
  · have hau : (a * u ·) =ᶠ[f] 0 := hu.mono fun x hx => by simp [hx]
    simp only [limsup_congr hu, limsup_congr hau, Pi.zero_def, ← ENNReal.bot_eq_zero,
      limsup_const_bot]
    /-
      case pos
      α : Type u_1
      f : Filter α
      inst✝ : CountableInterFilter f
      u : α → ENNReal
      a : ENNReal
      ha_top : Eq a Top.top
      hu : f.EventuallyEq u 0
      hau : f.EventuallyEq (fun x => HMul.hMul a (u x)) 0
      ⊢ Eq Bot.bot (HMul.hMul a Bot.bot)
    -/
    simp
    /-
      🎉 no goals
    -/
  · have hu_mul : ∃ᶠ x : α in f, ⊤ ≤ ite (u x = 0) (0 : ℝ≥0∞) ⊤ := by
      rw [EventuallyEq, not_eventually] at hu
      refine hu.mono fun x hx => ?_
      rw [Pi.zero_apply] at hx
      simp [hx]
    have h_top_le : (f.limsup fun x : α => ite (u x = 0) (0 : ℝ≥0∞) ⊤) = ⊤ :=
      eq_top_iff.mpr (le_limsup_of_frequently_le hu_mul)
    /-
      case neg
      α : Type u_1
      f : Filter α
      inst✝ : CountableInterFilter f
      u : α → ENNReal
      a : ENNReal
      ha_top : Eq a Top.top
      hu : Not (f.EventuallyEq u 0)
      hu_mul : Filter.Frequently (fun x => LE.le Top.top (ite (Eq (u x) 0) 0 Top.top …
      h_top_le : Eq (Filter.limsup (fun x => ite (Eq (u x) 0) 0 Top.top) f) Top.top
      ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
    -/
    have hfu : f.limsup u ≠ 0 := mt limsup_eq_zero_iff.1 hu
    /-
      case neg
      α : Type u_1
      f : Filter α
      inst✝ : CountableInterFilter f
      u : α → ENNReal
      a : ENNReal
      ha_top : Eq a Top.top
      hu : Not (f.EventuallyEq u 0)
      hu_mul : Filter.Frequently (fun x => LE.le Top.top (ite (Eq (u x) 0) 0 Top.top …
      h_top_le : Eq (Filter.limsup (fun x => ite (Eq (u x) 0) 0 Top.top) f) Top.top
      hfu : Ne (Filter.limsup u f) 0
      ⊢ Eq (Filter.limsup (fun x => HMul.hMul a (u x)) f) (HMul.hMul a (Filter.limsu …
    -/
    simp only [ha_top, top_mul', h_top_le, hfu, ite_false]
    /-
      🎉 no goals
    -/


/-- See also `limsup_mul_le'`.-/
theorem limsup_mul_le [CountableInterFilter f] (u v : α → ℝ≥0∞) :
    f.limsup (u * v) ≤ f.limsup u * f.limsup v :=
  calc
    f.limsup (u * v) ≤ f.limsup fun x => f.limsup u * v x := by
      /-
        α : Type u_1
        f : Filter α
        inst✝ : CountableInterFilter f
        u v : α → ENNReal
        ⊢ LE.le (Filter.limsup (HMul.hMul u v) f) (Filter.limsup (fun x => HMul.hMul ( …
      -/
      refine limsup_le_limsup ?_
      /-
        α : Type u_1
        f : Filter α
        inst✝ : CountableInterFilter f
        u v : α → ENNReal
        ⊢ f.EventuallyLE (HMul.hMul u v) fun x => HMul.hMul (Filter.limsup u f) (v x)
      -/
      filter_upwards [@eventually_le_limsup _ f _ u] with x hx using mul_le_mul' hx le_rfl
      /-
        🎉 no goals
      -/
    _ = f.limsup u * f.limsup v := limsup_const_mul


theorem limsup_add_le [CountableInterFilter f] (u v : α → ℝ≥0∞) :
    f.limsup (u + v) ≤ f.limsup u + f.limsup v :=
  sInf_le ((eventually_le_limsup u).mp
    ((eventually_le_limsup v).mono fun _ hxg hxf => add_le_add hxf hxg))


theorem limsup_liminf_le_liminf_limsup {β} [Countable β] {f : Filter α} [CountableInterFilter f]
    {g : Filter β} (u : α → β → ℝ≥0∞) :
    (f.limsup fun a : α => g.liminf fun b : β => u a b) ≤
      g.liminf fun b => f.limsup fun a => u a b :=
  have h1 : ∀ᶠ a in f, ∀ b, u a b ≤ f.limsup fun a' => u a' b := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Countable β
      f : Filter α
      inst✝ : CountableInterFilter f
      g : Filter β
      u : α → β → ENNReal
      ⊢ Filter.Eventually (fun a => ∀ (b : β), LE.le (u a b) (Filter.limsup (fun a'  …
    -/
    rw [eventually_countable_forall]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Countable β
      f : Filter α
      inst✝ : CountableInterFilter f
      g : Filter β
      u : α → β → ENNReal
      ⊢ ∀ (i : β), Filter.Eventually (fun x => LE.le (u x i) (Filter.limsup (fun a'  …
    -/
    exact fun b => ENNReal.eventually_le_limsup fun a => u a b
    /-
      🎉 no goals
    -/
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   inst✝¹ : Countable β
                                   f : Filter α
                                   inst✝ : CountableInterFilter f
                                   g : Filter β
                                   u : α → β → ENNReal
                                   h1 : Filter.Eventually (fun a => ∀ (b : β), LE.le (u a b) (Filter.limsup (fun  …
                                   x : α
                                   hx : ∀ (b : β), LE.le (u x b) (Filter.limsup (fun a' => u a' b) f)
                                   ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) g (u x)
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  sInf_le <| h1.mono fun x hx => Filter.liminf_le_liminf (Filter.Eventually.of_forall hx)
                                 /-
                                   🎉 no goals
                                 -/


