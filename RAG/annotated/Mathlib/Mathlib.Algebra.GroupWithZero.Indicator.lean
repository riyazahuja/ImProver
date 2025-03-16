lemma indicator_mul (s : Set ι) (f g : ι → M₀) :
    indicator s (fun i ↦ f i * g i) = fun i ↦ indicator s f i * indicator s g i := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s : Set ι
    f g : ι → M₀
    ⊢ Eq (s.indicator fun i => HMul.hMul (f i) (g i)) fun i => HMul.hMul (s.indica …
  -/
  funext
  /-
    case h
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s : Set ι
    f g : ι → M₀
    x✝ : ι
    ⊢ Eq (s.indicator (fun i => HMul.hMul (f i) (g i)) x✝) (HMul.hMul (s.indicator …
  -/
  simp only [indicator]
  /-
    case h
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s : Set ι
    f g : ι → M₀
    x✝ : ι
    ⊢ Eq (ite (Membership.mem s x✝) (HMul.hMul (f x✝) (g x✝)) 0) (HMul.hMul (ite ( …
  -/
  split_ifs
    /-
      case pos
      ι : Type u_1
      M₀ : Type u_4
      inst✝ : MulZeroClass M₀
      s : Set ι
      f g : ι → M₀
      x✝ : ι
      h✝ : Membership.mem s x✝
      ⊢ Eq (HMul.hMul (f x✝) (g x✝)) (HMul.hMul (f x✝) (g x✝))
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s : Set ι
    f g : ι → M₀
    x✝ : ι
    h✝ : Not (Membership.mem s x✝)
    ⊢ Eq 0 (HMul.hMul 0 0)
  -/
  rw [mul_zero]
  /-
    🎉 no goals
  -/


lemma indicator_mul_left (s : Set ι) (f g : ι → M₀) :
    indicator s (fun j ↦ f j * g j) i = indicator s f i * g i := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    i : ι
    s : Set ι
    f g : ι → M₀
    ⊢ Eq (s.indicator (fun j => HMul.hMul (f j) (g j)) i) (HMul.hMul (s.indicator  …
  -/
  simp only [indicator]
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    i : ι
    s : Set ι
    f g : ι → M₀
    ⊢ Eq (ite (Membership.mem s i) (HMul.hMul (f i) (g i)) 0) (HMul.hMul (ite (Mem …
  -/
  split_ifs
    /-
      case pos
      ι : Type u_1
      M₀ : Type u_4
      inst✝ : MulZeroClass M₀
      i : ι
      s : Set ι
      f g : ι → M₀
      h✝ : Membership.mem s i
      ⊢ Eq (HMul.hMul (f i) (g i)) (HMul.hMul (f i) (g i))
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      M₀ : Type u_4
      inst✝ : MulZeroClass M₀
      i : ι
      s : Set ι
      f g : ι → M₀
      h✝ : Not (Membership.mem s i)
      ⊢ Eq 0 (HMul.hMul 0 (g i))
    -/
  · rw [zero_mul]
    /-
      🎉 no goals
    -/


lemma indicator_mul_right (s : Set ι) (f g : ι → M₀) :
    indicator s (fun j ↦ f j * g j) i = f i * indicator s g i := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    i : ι
    s : Set ι
    f g : ι → M₀
    ⊢ Eq (s.indicator (fun j => HMul.hMul (f j) (g j)) i) (HMul.hMul (f i) (s.indi …
  -/
  simp only [indicator]
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    i : ι
    s : Set ι
    f g : ι → M₀
    ⊢ Eq (ite (Membership.mem s i) (HMul.hMul (f i) (g i)) 0) (HMul.hMul (f i) (it …
  -/
  split_ifs
    /-
      case pos
      ι : Type u_1
      M₀ : Type u_4
      inst✝ : MulZeroClass M₀
      i : ι
      s : Set ι
      f g : ι → M₀
      h✝ : Membership.mem s i
      ⊢ Eq (HMul.hMul (f i) (g i)) (HMul.hMul (f i) (g i))
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      M₀ : Type u_4
      inst✝ : MulZeroClass M₀
      i : ι
      s : Set ι
      f g : ι → M₀
      h✝ : Not (Membership.mem s i)
      ⊢ Eq 0 (HMul.hMul (f i) 0)
    -/
  · rw [mul_zero]
    /-
      🎉 no goals
    -/


lemma indicator_mul_const (s : Set ι) (f : ι → M₀) (a : M₀) (i : ι) :
                                                        /-
                                                          ι : Type u_1
                                                          M₀ : Type u_4
                                                          inst✝ : MulZeroClass M₀
                                                          s : Set ι
                                                          f : ι → M₀
                                                          a : M₀
                                                          i : ι
                                                          ⊢ Eq (s.indicator (fun x => HMul.hMul (f x) a) i) (HMul.hMul (s.indicator f i) …
                                                        -/
    s.indicator (f · * a) i = s.indicator f i * a := by rw [indicator_mul_left]
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma indicator_const_mul (s : Set ι) (f : ι → M₀) (a : M₀) (i : ι) :
                                                        /-
                                                          ι : Type u_1
                                                          M₀ : Type u_4
                                                          inst✝ : MulZeroClass M₀
                                                          s : Set ι
                                                          f : ι → M₀
                                                          a : M₀
                                                          i : ι
                                                          ⊢ Eq (s.indicator (fun x => HMul.hMul a (f x)) i) (HMul.hMul a (s.indicator f  …
                                                        -/
    s.indicator (a * f ·) i = a * s.indicator f i := by rw [indicator_mul_right]
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma inter_indicator_mul (f g : ι → M₀) (i : ι) :
    (s ∩ t).indicator (fun j ↦ f j * g j) i = s.indicator f i * t.indicator g i := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s t : Set ι
    f g : ι → M₀
    i : ι
    ⊢ Eq ((Inter.inter s t).indicator (fun j => HMul.hMul (f j) (g j)) i) (HMul.hM …
  -/
  rw [← Set.indicator_indicator]
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s t : Set ι
    f g : ι → M₀
    i : ι
    ⊢ Eq (s.indicator (t.indicator fun j => HMul.hMul (f j) (g j)) i) (HMul.hMul ( …
  -/
  simp_rw [indicator]
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : MulZeroClass M₀
    s t : Set ι
    f g : ι → M₀
    i : ι
    ⊢ Eq (ite (Membership.mem s i) (ite (Membership.mem t i) (HMul.hMul (f i) (g i …
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
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


lemma inter_indicator_one : (s ∩ t).indicator (1 : ι → M₀) = s.indicator 1 * t.indicator 1 :=
                    /-
                      ι : Type u_1
                      M₀ : Type u_4
                      inst✝ : MulZeroOneClass M₀
                      s t : Set ι
                      x✝ : ι
                      ⊢ Eq ((Inter.inter s t).indicator 1 x✝) (HMul.hMul (s.indicator 1) (t.indicato …
                    -/
  funext fun _ ↦ by simp only [← inter_indicator_mul, Pi.mul_apply, Pi.one_apply, one_mul]; congr
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


lemma indicator_prod_one {t : Set κ} {j : κ} :
    (s ×ˢ t).indicator (1 : ι × κ → M₀) (i, j) = s.indicator 1 i * t.indicator 1 j := by
  /-
    ι : Type u_1
    κ : Type u_2
    M₀ : Type u_4
    inst✝ : MulZeroOneClass M₀
    s : Set ι
    i : ι
    t : Set κ
    j : κ
    ⊢ Eq ((SProd.sprod s t).indicator 1 { fst := i, snd := j }) (HMul.hMul (s.indi …
  -/
  simp_rw [indicator, mem_prod_eq]
  /-
    ι : Type u_1
    κ : Type u_2
    M₀ : Type u_4
    inst✝ : MulZeroOneClass M₀
    s : Set ι
    i : ι
    t : Set κ
    j : κ
    ⊢ Eq (ite (And (Membership.mem s i) (Membership.mem t j)) (1 { fst := i, snd : …
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
                          🎉 no goals
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
  split_ifs with h₀ <;> simp only [Pi.one_apply, mul_one, mul_zero] <;> tauto
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


lemma indicator_eq_zero_iff_not_mem : indicator s 1 i = (0 : M₀) ↔ i ∉ s := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝¹ : MulZeroOneClass M₀
    s : Set ι
    i : ι
    inst✝ : Nontrivial M₀
    ⊢ Iff (Eq (s.indicator 1 i) 0) (Not (Membership.mem s i))
  -/
  classical simp [indicator_apply, imp_false]
  /-
    🎉 no goals
  -/


lemma indicator_eq_one_iff_mem : indicator s 1 i = (1 : M₀) ↔ i ∈ s := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝¹ : MulZeroOneClass M₀
    s : Set ι
    i : ι
    inst✝ : Nontrivial M₀
    ⊢ Iff (Eq (s.indicator 1 i) 1) (Membership.mem s i)
  -/
  classical simp [indicator_apply, imp_false]
  /-
    🎉 no goals
  -/


lemma indicator_one_inj (h : indicator s (1 : ι → M₀) = indicator t 1) : s = t := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝¹ : MulZeroOneClass M₀
    s t : Set ι
    inst✝ : Nontrivial M₀
    h : Eq (s.indicator 1) (t.indicator 1)
    ⊢ Eq s t
  -/
  ext; simp_rw [← indicator_eq_one_iff_mem M₀, h]
       /-
         🎉 no goals
       -/


@[simp] lemma support_one : support (1 : ι → R) = univ := support_const one_ne_zero


@[simp] lemma mulSupport_zero : mulSupport (0 : ι → R) = univ := mulSupport_const zero_ne_one


lemma support_mul_subset_left (f g : ι → M₀) : support (fun x ↦ f x * g x) ⊆ support f :=
                           /-
                             ι : Type u_1
                             M₀ : Type u_4
                             inst✝ : MulZeroClass M₀
                             f g : ι → M₀
                             x : ι
                             hfg : Membership.mem (Function.support fun x => HMul.hMul (f x) (g x)) x
                             hf : Eq (f x) 0
                             ⊢ Eq ((fun x => HMul.hMul (f x) (g x)) x) 0
                           -/
  fun x hfg hf ↦ hfg <| by simp only [hf, zero_mul]
                           /-
                             🎉 no goals
                           -/

--@[simp] Porting note: removing simp, bad lemma LHS not in normal form

lemma support_mul_subset_right (f g : ι → M₀) : support (fun x ↦ f x * g x) ⊆ support g :=
                            /-
                              ι : Type u_1
                              M₀ : Type u_4
                              inst✝ : MulZeroClass M₀
                              f g : ι → M₀
                              x : ι
                              hfg : Membership.mem (Function.support fun x => HMul.hMul (f x) (g x)) x
                              hg : Eq (g x) 0
                              ⊢ Eq ((fun x => HMul.hMul (f x) (g x)) x) 0
                            -/
  fun x hfg hg => hfg <| by simp only [hg, mul_zero]
                            /-
                              🎉 no goals
                            -/


@[simp] lemma support_mul (f g : ι → M₀) : support (fun x ↦ f x * g x) = support f ∩ support g :=
                 /-
                   ι : Type u_1
                   M₀ : Type u_4
                   inst✝¹ : MulZeroClass M₀
                   inst✝ : NoZeroDivisors M₀
                   f g : ι → M₀
                   x : ι
                   ⊢ Iff (Membership.mem (Function.support fun x => HMul.hMul (f x) (g x)) x) (Me …
                 -/
  ext fun x ↦ by simp [not_or]
                 /-
                   🎉 no goals
                 -/


@[simp] lemma support_mul' (f g : ι → M₀) : support (f * g) = support f ∩ support g :=
  support_mul _ _


@[simp] lemma support_pow (f : ι → M₀) (hn : n ≠ 0) : support (fun a ↦ f a ^ n) = support f := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝¹ : MonoidWithZero M₀
    inst✝ : NoZeroDivisors M₀
    n : Nat
    f : ι → M₀
    hn : Ne n 0
    ⊢ Eq (Function.support fun a => HPow.hPow (f a) n) (Function.support f)
  -/
  ext; exact (pow_eq_zero_iff hn).not
       /-
         🎉 no goals
       -/


@[simp] lemma support_pow' (f : ι → M₀) (hn : n ≠ 0) : support (f ^ n) = support f :=
  support_pow _ hn


@[simp] lemma support_inv (f : ι → G₀) : support (fun a ↦ (f a)⁻¹) = support f :=
  Set.ext fun _ ↦ not_congr inv_eq_zero


@[simp] lemma support_inv' (f : ι → G₀) : support f⁻¹ = support f := support_inv _


@[simp] lemma support_div (f g : ι → G₀) : support (fun a ↦ f a / g a) = support f ∩ support g := by
  /-
    ι : Type u_1
    G₀ : Type u_3
    inst✝ : GroupWithZero G₀
    f g : ι → G₀
    ⊢ Eq (Function.support fun a => HDiv.hDiv (f a) (g a)) (Inter.inter (Function. …
  -/
  simp [div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp] lemma support_div' (f g : ι → G₀) : support (f / g) = support f ∩ support g :=
  support_div _ _


lemma mulSupport_one_add [AddLeftCancelMonoid R] (f : ι → R) :
    mulSupport (fun x ↦ 1 + f x) = support f :=
  Set.ext fun _ ↦ not_congr add_right_eq_self


lemma mulSupport_one_add' [AddLeftCancelMonoid R] (f : ι → R) : mulSupport (1 + f) = support f :=
  mulSupport_one_add f


lemma mulSupport_add_one [AddRightCancelMonoid R] (f : ι → R) :
    mulSupport (fun x ↦ f x + 1) = support f := Set.ext fun _ ↦ not_congr add_left_eq_self


lemma mulSupport_add_one' [AddRightCancelMonoid R] (f : ι → R) : mulSupport (f + 1) = support f :=
  mulSupport_add_one f


lemma mulSupport_one_sub' [AddGroup R] (f : ι → R) : mulSupport (1 - f) = support f := by
  /-
    ι : Type u_1
    R : Type u_5
    inst✝¹ : One R
    inst✝ : AddGroup R
    f : ι → R
    ⊢ Eq (Function.mulSupport (HSub.hSub 1 f)) (Function.support f)
  -/
  rw [sub_eq_add_neg, mulSupport_one_add', support_neg']
  /-
    🎉 no goals
  -/


lemma mulSupport_one_sub [AddGroup R] (f : ι → R) :
    mulSupport (fun x ↦ 1 - f x) = support f := mulSupport_one_sub' f


