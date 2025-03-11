/-- `f` has superpolynomial decay in parameter `k` along filter `l` if
  `k ^ n * f` tends to zero at `l` for all naturals `n` -/
def SuperpolynomialDecay {α β : Type*} [TopologicalSpace β] [CommSemiring β] (l : Filter α)
    (k : α → β) (f : α → β) :=
  ∀ n : ℕ, Tendsto (fun a : α => k a ^ n * f a) l (𝓝 0)


theorem SuperpolynomialDecay.congr' (hf : SuperpolynomialDecay l k f) (hfg : f =ᶠ[l] g) :
    SuperpolynomialDecay l k g := fun z =>
  (hf z).congr' (EventuallyEq.mul (EventuallyEq.refl l _) hfg)


theorem SuperpolynomialDecay.congr (hf : SuperpolynomialDecay l k f) (hfg : ∀ x, f x = g x) :
    SuperpolynomialDecay l k g := fun z =>
  (hf z).congr fun x => (congr_arg fun a => k x ^ z * a) <| hfg x


@[simp]
theorem superpolynomialDecay_zero (l : Filter α) (k : α → β) : SuperpolynomialDecay l k 0 :=
              /-
                α : Type u_1
                β : Type u_2
                inst✝¹ : TopologicalSpace β
                inst✝ : CommSemiring β
                l : Filter α
                k : α → β
                z : Nat
                ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (0 a)) l (nhds 0)
              -/
  fun z => by simpa only [Pi.zero_apply, mul_zero] using tendsto_const_nhds
              /-
                🎉 no goals
              -/


theorem SuperpolynomialDecay.add [ContinuousAdd β] (hf : SuperpolynomialDecay l k f)
    (hg : SuperpolynomialDecay l k g) : SuperpolynomialDecay l k (f + g) := fun z => by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f g : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : CommSemiring β
    inst✝ : ContinuousAdd β
    hf : Asymptotics.SuperpolynomialDecay l k f
    hg : Asymptotics.SuperpolynomialDecay l k g
    z : Nat
    ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (HAdd.hAdd f g a)) l  …
  -/
  simpa only [mul_add, add_zero, Pi.add_apply] using (hf z).add (hg z)
  /-
    🎉 no goals
  -/


theorem SuperpolynomialDecay.mul [ContinuousMul β] (hf : SuperpolynomialDecay l k f)
    (hg : SuperpolynomialDecay l k g) : SuperpolynomialDecay l k (f * g) := fun z => by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f g : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : CommSemiring β
    inst✝ : ContinuousMul β
    hf : Asymptotics.SuperpolynomialDecay l k f
    hg : Asymptotics.SuperpolynomialDecay l k g
    z : Nat
    ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (HMul.hMul f g a)) l  …
  -/
  simpa only [mul_assoc, one_mul, mul_zero, pow_zero] using (hf z).mul (hg 0)
  /-
    🎉 no goals
  -/


theorem SuperpolynomialDecay.mul_const [ContinuousMul β] (hf : SuperpolynomialDecay l k f) (c : β) :
    SuperpolynomialDecay l k fun n => f n * c := fun z => by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : CommSemiring β
    inst✝ : ContinuousMul β
    hf : Asymptotics.SuperpolynomialDecay l k f
    c : β
    z : Nat
    ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) ((fun n => HMul.hMul  …
  -/
  simpa only [← mul_assoc, zero_mul] using Tendsto.mul_const c (hf z)
  /-
    🎉 no goals
  -/


theorem SuperpolynomialDecay.const_mul [ContinuousMul β] (hf : SuperpolynomialDecay l k f) (c : β) :
    SuperpolynomialDecay l k fun n => c * f n :=
  (hf.mul_const c).congr fun _ => mul_comm _ _


theorem SuperpolynomialDecay.param_mul (hf : SuperpolynomialDecay l k f) :
    SuperpolynomialDecay l k (k * f) := fun z =>
  tendsto_nhds.2 fun s hs hs0 =>
    l.sets_of_superset ((tendsto_nhds.1 (hf <| z + 1)) s hs hs0) fun x hx => by
      /-
        α : Type u_1
        β : Type u_2
        l : Filter α
        k f : α → β
        inst✝¹ : TopologicalSpace β
        inst✝ : CommSemiring β
        hf : Asymptotics.SuperpolynomialDecay l k f
        z : Nat
        s : Set β
        hs : IsOpen s
        hs0 : Membership.mem s 0
        x : α
        hx : Membership.mem (Set.preimage (fun a => HMul.hMul (HPow.hPow (k a) (HAdd.h …
        ⊢ Membership.mem (Set.preimage (fun a => HMul.hMul (HPow.hPow (k a) z) (HMul.h …
      -/
      simpa only [Set.mem_preimage, Pi.mul_apply, ← mul_assoc, ← pow_succ] using hx
      /-
        🎉 no goals
      -/


theorem SuperpolynomialDecay.mul_param (hf : SuperpolynomialDecay l k f) :
    SuperpolynomialDecay l k (f * k) :=
  hf.param_mul.congr fun _ => mul_comm _ _


theorem SuperpolynomialDecay.param_pow_mul (hf : SuperpolynomialDecay l k f) (n : ℕ) :
    SuperpolynomialDecay l k (k ^ n * f) := by
  induction n with
  | zero => simpa only [one_mul, pow_zero] using hf
  | succ n hn => simpa only [pow_succ', mul_assoc] using hn.param_mul


theorem SuperpolynomialDecay.mul_param_pow (hf : SuperpolynomialDecay l k f) (n : ℕ) :
    SuperpolynomialDecay l k (f * k ^ n) :=
  (hf.param_pow_mul n).congr fun _ => mul_comm _ _


theorem SuperpolynomialDecay.polynomial_mul [ContinuousAdd β] [ContinuousMul β]
    (hf : SuperpolynomialDecay l k f) (p : β[X]) :
    SuperpolynomialDecay l k fun x => (p.eval <| k x) * f x :=
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    l : Filter α
                                                    k f : α → β
                                                    inst✝³ : TopologicalSpace β
                                                    inst✝² : CommSemiring β
                                                    inst✝¹ : ContinuousAdd β
                                                    inst✝ : ContinuousMul β
                                                    hf : Asymptotics.SuperpolynomialDecay l k f
                                                    p✝ p q : Polynomial β
                                                    hp : Asymptotics.SuperpolynomialDecay l k fun x => HMul.hMul (Polynomial.eval  …
                                                    hq : Asymptotics.SuperpolynomialDecay l k fun x => HMul.hMul (Polynomial.eval  …
                                                    ⊢ Asymptotics.SuperpolynomialDecay l k fun x => HMul.hMul (Polynomial.eval (k  …
                                                  -/
  Polynomial.induction_on' p (fun p q hp hq => by simpa [add_mul] using hp.add hq) fun n c => by
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝³ : TopologicalSpace β
      inst✝² : CommSemiring β
      inst✝¹ : ContinuousAdd β
      inst✝ : ContinuousMul β
      hf : Asymptotics.SuperpolynomialDecay l k f
      p : Polynomial β
      n : Nat
      c : β
      ⊢ Asymptotics.SuperpolynomialDecay l k fun x => HMul.hMul (Polynomial.eval (k  …
    -/
    simpa [mul_assoc] using (hf.param_pow_mul n).const_mul c
    /-
      🎉 no goals
    -/


theorem SuperpolynomialDecay.mul_polynomial [ContinuousAdd β] [ContinuousMul β]
    (hf : SuperpolynomialDecay l k f) (p : β[X]) :
    SuperpolynomialDecay l k fun x => f x * (p.eval <| k x) :=
  (hf.polynomial_mul p).congr fun _ => mul_comm _ _


theorem SuperpolynomialDecay.trans_eventuallyLE (hk : 0 ≤ᶠ[l] k) (hg : SuperpolynomialDecay l k g)
    (hg' : SuperpolynomialDecay l k g') (hfg : g ≤ᶠ[l] f) (hfg' : f ≤ᶠ[l] g') :
    SuperpolynomialDecay l k f := fun z =>
  tendsto_of_tendsto_of_tendsto_of_le_of_le' (hg z) (hg' z)
    (hfg.mp (hk.mono fun _ hx hx' => mul_le_mul_of_nonneg_left hx' (pow_nonneg hx z)))
    (hfg'.mp (hk.mono fun _ hx hx' => mul_le_mul_of_nonneg_left hx' (pow_nonneg hx z)))


theorem superpolynomialDecay_iff_abs_tendsto_zero :
    SuperpolynomialDecay l k f ↔ ∀ n : ℕ, Tendsto (fun a : α => |k a ^ n * f a|) l (𝓝 0) :=
  ⟨fun h z => (tendsto_zero_iff_abs_tendsto_zero _).1 (h z), fun h z =>
    (tendsto_zero_iff_abs_tendsto_zero _).2 (h z)⟩


theorem superpolynomialDecay_iff_superpolynomialDecay_abs :
    SuperpolynomialDecay l k f ↔ SuperpolynomialDecay l (fun a => |k a|) fun a => |f a| :=
  (superpolynomialDecay_iff_abs_tendsto_zero l k f).trans
        /-
          α : Type u_1
          β : Type u_2
          l : Filter α
          k f : α → β
          inst✝² : TopologicalSpace β
          inst✝¹ : LinearOrderedCommRing β
          inst✝ : OrderTopology β
          ⊢ Iff (∀ (n : Nat), Filter.Tendsto (fun a => abs (HMul.hMul (HPow.hPow (k a) n …
        -/
    (by simp_rw [SuperpolynomialDecay, abs_mul, abs_pow])
        /-
          🎉 no goals
        -/


theorem SuperpolynomialDecay.trans_eventually_abs_le (hf : SuperpolynomialDecay l k f)
    (hfg : abs ∘ g ≤ᶠ[l] abs ∘ f) : SuperpolynomialDecay l k g := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f g : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedCommRing β
    inst✝ : OrderTopology β
    hf : Asymptotics.SuperpolynomialDecay l k f
    hfg : l.EventuallyLE (Function.comp abs g) (Function.comp abs f)
    ⊢ Asymptotics.SuperpolynomialDecay l k g
  -/
  rw [superpolynomialDecay_iff_abs_tendsto_zero] at hf ⊢
  refine fun z =>
    tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds (hf z)
      (Eventually.of_forall fun x => abs_nonneg _) (hfg.mono fun x hx => ?_)
  calc
    |k x ^ z * g x| = |k x ^ z| * |g x| := abs_mul (k x ^ z) (g x)
    _ ≤ |k x ^ z| * |f x| := by gcongr _ * ?_; exact hx
    _ = |k x ^ z * f x| := (abs_mul (k x ^ z) (f x)).symm


theorem SuperpolynomialDecay.trans_abs_le (hf : SuperpolynomialDecay l k f)
    (hfg : ∀ x, |g x| ≤ |f x|) : SuperpolynomialDecay l k g :=
  hf.trans_eventually_abs_le (Eventually.of_forall hfg)


theorem superpolynomialDecay_mul_const_iff [ContinuousMul β] {c : β} (hc0 : c ≠ 0) :
    (SuperpolynomialDecay l k fun n => f n * c) ↔ SuperpolynomialDecay l k f :=
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  l : Filter α
                                                  k f : α → β
                                                  inst✝² : TopologicalSpace β
                                                  inst✝¹ : Field β
                                                  inst✝ : ContinuousMul β
                                                  c : β
                                                  hc0 : Ne c 0
                                                  h : Asymptotics.SuperpolynomialDecay l k fun n => HMul.hMul (f n) c
                                                  x : α
                                                  ⊢ Eq (HMul.hMul (HMul.hMul (f x) c) (Inv.inv c)) (f x)
                                                -/
  ⟨fun h => (h.mul_const c⁻¹).congr fun x => by simp [mul_assoc, mul_inv_cancel₀ hc0], fun h =>
                                                /-
                                                  🎉 no goals
                                                -/
    h.mul_const c⟩


theorem superpolynomialDecay_const_mul_iff [ContinuousMul β] {c : β} (hc0 : c ≠ 0) :
    (SuperpolynomialDecay l k fun n => c * f n) ↔ SuperpolynomialDecay l k f :=
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  l : Filter α
                                                  k f : α → β
                                                  inst✝² : TopologicalSpace β
                                                  inst✝¹ : Field β
                                                  inst✝ : ContinuousMul β
                                                  c : β
                                                  hc0 : Ne c 0
                                                  h : Asymptotics.SuperpolynomialDecay l k fun n => HMul.hMul c (f n)
                                                  x : α
                                                  ⊢ Eq (HMul.hMul (Inv.inv c) (HMul.hMul c (f x))) (f x)
                                                -/
  ⟨fun h => (h.const_mul c⁻¹).congr fun x => by simp [← mul_assoc, inv_mul_cancel₀ hc0], fun h =>
                                                /-
                                                  🎉 no goals
                                                -/
    h.const_mul c⟩


theorem superpolynomialDecay_iff_abs_isBoundedUnder (hk : Tendsto k l atTop) :
    SuperpolynomialDecay l k f ↔
    ∀ z : ℕ, IsBoundedUnder (· ≤ ·) l fun a : α => |k a ^ z * f a| := by
  refine
    ⟨fun h z => Tendsto.isBoundedUnder_le (Tendsto.abs (h z)), fun h =>
      (superpolynomialDecay_iff_abs_tendsto_zero l k f).2 fun z => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : ∀ (z : Nat), Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun a => a …
    z : Nat
    ⊢ Filter.Tendsto (fun a => abs (HMul.hMul (HPow.hPow (k a) z) (f a))) l (nhds 0)
  -/
  obtain ⟨m, hm⟩ := h (z + 1)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : ∀ (z : Nat), Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun a => a …
    z : Nat
    m : β
    hm : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x m) (Filter.map ( …
    ⊢ Filter.Tendsto (fun a => abs (HMul.hMul (HPow.hPow (k a) z) (f a))) l (nhds 0)
  -/
  have h1 : Tendsto (fun _ : α => (0 : β)) l (𝓝 0) := tendsto_const_nhds
  have h2 : Tendsto (fun a : α => |(k a)⁻¹| * m) l (𝓝 0) :=
    zero_mul m ▸
      Tendsto.mul_const m ((tendsto_zero_iff_abs_tendsto_zero _).1 hk.inv_tendsto_atTop)
  refine
    tendsto_of_tendsto_of_tendsto_of_le_of_le' h1 h2 (Eventually.of_forall fun x => abs_nonneg _)
      ((eventually_map.1 hm).mp ?_)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : ∀ (z : Nat), Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun a => a …
    z : Nat
    m : β
    hm : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x m) (Filter.map ( …
    h1 : Filter.Tendsto (fun x => 0) l (nhds 0)
    h2 : Filter.Tendsto (fun a => HMul.hMul (abs (Inv.inv (k a))) m) l (nhds 0)
    ⊢ Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) (abs (HMul.hMul (HPow …
  -/
  refine (hk.eventually_ne_atTop 0).mono fun x hk0 hx => ?_
  /-
    case intro
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : ∀ (z : Nat), Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun a => a …
    z : Nat
    m : β
    hm : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x m) (Filter.map ( …
    h1 : Filter.Tendsto (fun x => 0) l (nhds 0)
    h2 : Filter.Tendsto (fun a => HMul.hMul (abs (Inv.inv (k a))) m) l (nhds 0)
    x : α
    hk0 : Ne (k x) 0
    hx : (fun x1 x2 => LE.le x1 x2) (abs (HMul.hMul (HPow.hPow (k x) (HAdd.hAdd z  …
    ⊢ LE.le (abs (HMul.hMul (HPow.hPow (k x) z) (f x))) (HMul.hMul (abs (Inv.inv ( …
  -/
  refine Eq.trans_le ?_ (mul_le_mul_of_nonneg_left hx <| abs_nonneg (k x)⁻¹)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : ∀ (z : Nat), Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun a => a …
    z : Nat
    m : β
    hm : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x m) (Filter.map ( …
    h1 : Filter.Tendsto (fun x => 0) l (nhds 0)
    h2 : Filter.Tendsto (fun a => HMul.hMul (abs (Inv.inv (k a))) m) l (nhds 0)
    x : α
    hk0 : Ne (k x) 0
    hx : (fun x1 x2 => LE.le x1 x2) (abs (HMul.hMul (HPow.hPow (k x) (HAdd.hAdd z  …
    ⊢ Eq (abs (HMul.hMul (HPow.hPow (k x) z) (f x))) (HMul.hMul (abs (Inv.inv (k x …
  -/
  rw [← abs_mul, ← mul_assoc, pow_succ', ← mul_assoc, inv_mul_cancel₀ hk0, one_mul]
  /-
    🎉 no goals
  -/


theorem superpolynomialDecay_iff_zpow_tendsto_zero (hk : Tendsto k l atTop) :
    SuperpolynomialDecay l k f ↔ ∀ z : ℤ, Tendsto (fun a : α => k a ^ z * f a) l (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    ⊢ Iff (Asymptotics.SuperpolynomialDecay l k f) (∀ (z : Int), Filter.Tendsto (f …
  -/
  refine ⟨fun h z => ?_, fun h n => by simpa only [zpow_natCast] using h (n : ℤ)⟩
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : Asymptotics.SuperpolynomialDecay l k f
    z : Int
    ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) l (nhds 0)
  -/
  by_cases hz : 0 ≤ z
    /-
      case pos
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : LinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      h : Asymptotics.SuperpolynomialDecay l k f
      z : Int
      hz : LE.le 0 z
      ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) l (nhds 0)
    -/
  · unfold Tendsto
    /-
      case pos
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : LinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      h : Asymptotics.SuperpolynomialDecay l k f
      z : Int
      hz : LE.le 0 z
      ⊢ LE.le (Filter.map (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) l) (nhds 0)
    -/
    lift z to ℕ using hz
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : LinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      h : Asymptotics.SuperpolynomialDecay l k f
      z : Nat
      ⊢ LE.le (Filter.map (fun a => HMul.hMul (HPow.hPow (k a) ↑z) (f a)) l) (nhds 0)
    -/
    simpa using h z
    /-
      🎉 no goals
    -/
  · have : Tendsto (fun a => k a ^ z) l (𝓝 0) :=
      Tendsto.comp (tendsto_zpow_atTop_zero (not_le.1 hz)) hk
    /-
      case neg
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : LinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      h : Asymptotics.SuperpolynomialDecay l k f
      z : Int
      hz : Not (LE.le 0 z)
      this : Filter.Tendsto (fun a => HPow.hPow (k a) z) l (nhds 0)
      ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) l (nhds 0)
    -/
    have h : Tendsto f l (𝓝 0) := by simpa using h 0
    /-
      case neg
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : LinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      h✝ : Asymptotics.SuperpolynomialDecay l k f
      z : Int
      hz : Not (LE.le 0 z)
      this : Filter.Tendsto (fun a => HPow.hPow (k a) z) l (nhds 0)
      h : Filter.Tendsto f l (nhds 0)
      ⊢ Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) l (nhds 0)
    -/
    exact zero_mul (0 : β) ▸ this.mul h
    /-
      🎉 no goals
    -/


theorem SuperpolynomialDecay.param_zpow_mul (hk : Tendsto k l atTop)
    (hf : SuperpolynomialDecay l k f) (z : ℤ) :
    SuperpolynomialDecay l k fun a => k a ^ z * f a := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    hf : Asymptotics.SuperpolynomialDecay l k f
    z : Int
    ⊢ Asymptotics.SuperpolynomialDecay l k fun a => HMul.hMul (HPow.hPow (k a) z)  …
  -/
  rw [superpolynomialDecay_iff_zpow_tendsto_zero _ hk] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    hf : ∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a) …
    z : Int
    ⊢ ∀ (z_1 : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z_1) (HMu …
  -/
  refine fun z' => (hf <| z' + z).congr' ((hk.eventually_ne_atTop 0).mono fun x hx => ?_)
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    hf : ∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a) …
    z z' : Int
    x : α
    hx : Ne (k x) 0
    ⊢ Eq (HMul.hMul (HPow.hPow (k x) (HAdd.hAdd z' z)) (f x)) ((fun a => HMul.hMul …
  -/
  simp [zpow_add₀ hx, mul_assoc, Pi.mul_apply]
  /-
    🎉 no goals
  -/


theorem SuperpolynomialDecay.mul_param_zpow (hk : Tendsto k l atTop)
    (hf : SuperpolynomialDecay l k f) (z : ℤ) : SuperpolynomialDecay l k fun a => f a * k a ^ z :=
  (hf.param_zpow_mul hk z).congr fun _ => mul_comm _ _


theorem SuperpolynomialDecay.inv_param_mul (hk : Tendsto k l atTop)
    (hf : SuperpolynomialDecay l k f) : SuperpolynomialDecay l k (k⁻¹ * f) := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    hf : Asymptotics.SuperpolynomialDecay l k f
    ⊢ Asymptotics.SuperpolynomialDecay l k (HMul.hMul (Inv.inv k) f)
  -/
  simpa using hf.param_zpow_mul hk (-1)
  /-
    🎉 no goals
  -/


theorem SuperpolynomialDecay.param_inv_mul (hk : Tendsto k l atTop)
    (hf : SuperpolynomialDecay l k f) : SuperpolynomialDecay l k (f * k⁻¹) :=
  (hf.inv_param_mul hk).congr fun _ => mul_comm _ _


theorem superpolynomialDecay_param_mul_iff (hk : Tendsto k l atTop) :
    SuperpolynomialDecay l k (k * f) ↔ SuperpolynomialDecay l k f :=
  ⟨fun h =>
    (h.inv_param_mul hk).congr'
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        l : Filter α
                                                        k f : α → β
                                                        inst✝² : TopologicalSpace β
                                                        inst✝¹ : LinearOrderedField β
                                                        inst✝ : OrderTopology β
                                                        hk : Filter.Tendsto k l Filter.atTop
                                                        h : Asymptotics.SuperpolynomialDecay l k (HMul.hMul k f)
                                                        x : α
                                                        hx : Ne (k x) 0
                                                        ⊢ Eq (HMul.hMul (Inv.inv k) (HMul.hMul k f) x) (f x)
                                                      -/
      ((hk.eventually_ne_atTop 0).mono fun x hx => by simp [← mul_assoc, inv_mul_cancel₀ hx]),
                                                      /-
                                                        🎉 no goals
                                                      -/
    fun h => h.param_mul⟩


theorem superpolynomialDecay_mul_param_iff (hk : Tendsto k l atTop) :
    SuperpolynomialDecay l k (f * k) ↔ SuperpolynomialDecay l k f := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    ⊢ Iff (Asymptotics.SuperpolynomialDecay l k (HMul.hMul f k)) (Asymptotics.Supe …
  -/
  simpa [mul_comm k] using superpolynomialDecay_param_mul_iff f hk
  /-
    🎉 no goals
  -/


theorem superpolynomialDecay_param_pow_mul_iff (hk : Tendsto k l atTop) (n : ℕ) :
    SuperpolynomialDecay l k (k ^ n * f) ↔ SuperpolynomialDecay l k f := by
  induction n with
  | zero => simp
  | succ n hn =>
    simpa [pow_succ, ← mul_comm k, mul_assoc,
      superpolynomialDecay_param_mul_iff (k ^ n * f) hk] using hn


theorem superpolynomialDecay_mul_param_pow_iff (hk : Tendsto k l atTop) (n : ℕ) :
    SuperpolynomialDecay l k (f * k ^ n) ↔ SuperpolynomialDecay l k f := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    n : Nat
    ⊢ Iff (Asymptotics.SuperpolynomialDecay l k (HMul.hMul f (HPow.hPow k n))) (As …
  -/
  simpa [mul_comm f] using superpolynomialDecay_param_pow_mul_iff f hk n
  /-
    🎉 no goals
  -/


theorem superpolynomialDecay_iff_norm_tendsto_zero :
    SuperpolynomialDecay l k f ↔ ∀ n : ℕ, Tendsto (fun a : α => ‖k a ^ n * f a‖) l (𝓝 0) :=
  ⟨fun h z => tendsto_zero_iff_norm_tendsto_zero.1 (h z), fun h z =>
    tendsto_zero_iff_norm_tendsto_zero.2 (h z)⟩


theorem superpolynomialDecay_iff_superpolynomialDecay_norm :
    SuperpolynomialDecay l k f ↔ SuperpolynomialDecay l (fun a => ‖k a‖) fun a => ‖f a‖ :=
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 l : Filter α
                                                                 k f : α → β
                                                                 inst✝ : NormedLinearOrderedField β
                                                                 ⊢ Iff (∀ (n : Nat), Filter.Tendsto (fun a => Norm.norm (HMul.hMul (HPow.hPow ( …
                                                               -/
  (superpolynomialDecay_iff_norm_tendsto_zero l k f).trans (by simp [SuperpolynomialDecay])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem superpolynomialDecay_iff_isBigO (hk : Tendsto k l atTop) :
    SuperpolynomialDecay l k f ↔ ∀ z : ℤ, f =O[l] fun a : α => k a ^ z := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    ⊢ Iff (Asymptotics.SuperpolynomialDecay l k f) (∀ (z : Int), Asymptotics.IsBig …
  -/
  refine (superpolynomialDecay_iff_zpow_tendsto_zero f hk).trans ?_
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    ⊢ Iff (∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f  …
  -/
  have hk0 : ∀ᶠ x in l, k x ≠ 0 := hk.eventually_ne_atTop 0
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
    ⊢ Iff (∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f  …
  -/
  refine ⟨fun h z => ?_, fun h z => ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝¹ : NormedLinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
      h : ∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) …
      z : Int
      ⊢ Asymptotics.IsBigO l f fun a => HPow.hPow (k a) z
    -/
  · refine isBigO_of_div_tendsto_nhds (hk0.mono fun x hx hxz ↦ absurd hxz (zpow_ne_zero _ hx)) 0 ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝¹ : NormedLinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
      h : ∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) …
      z : Int
      ⊢ Filter.Tendsto (HDiv.hDiv f fun a => HPow.hPow (k a) z) l (nhds 0)
    -/
    have : (fun a : α => k a ^ z)⁻¹ = fun a : α => k a ^ (-z) := funext fun x => by simp
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝¹ : NormedLinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
      h : ∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) …
      z : Int
      this : Eq (Inv.inv fun a => HPow.hPow (k a) z) fun a => HPow.hPow (k a) (Neg.n …
      ⊢ Filter.Tendsto (HDiv.hDiv f fun a => HPow.hPow (k a) z) l (nhds 0)
    -/
    rw [div_eq_mul_inv, mul_comm f, this]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝¹ : NormedLinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
      h : ∀ (z : Int), Filter.Tendsto (fun a => HMul.hMul (HPow.hPow (k a) z) (f a)) …
      z : Int
      this : Eq (Inv.inv fun a => HPow.hPow (k a) z) fun a => HPow.hPow (k a) (Neg.n …
      ⊢ Filter.Tendsto (HMul.hMul (fun a => HPow.hPow (k a) (Neg.neg z)) f) l (nhds 0)
    -/
    exact h (-z)
    /-
      🎉 no goals
    -/
  · suffices (fun a : α => k a ^ z * f a) =O[l] fun a : α => (k a)⁻¹ from
      IsBigO.trans_tendsto this hk.inv_tendsto_atTop
    refine
      ((isBigO_refl (fun a => k a ^ z) l).mul (h (-(z + 1)))).trans
        (IsBigO.of_bound 1 <| hk0.mono fun a ha0 => ?_)
    simp only [one_mul, neg_add z 1, zpow_add₀ ha0, ← mul_assoc, zpow_neg,
      mul_inv_cancel₀ (zpow_ne_zero z ha0), zpow_one]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      l : Filter α
      k f : α → β
      inst✝¹ : NormedLinearOrderedField β
      inst✝ : OrderTopology β
      hk : Filter.Tendsto k l Filter.atTop
      hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
      h : ∀ (z : Int), Asymptotics.IsBigO l f fun a => HPow.hPow (k a) z
      z : Int
      a : α
      ha0 : Ne (k a) 0
      ⊢ LE.le (Norm.norm (Inv.inv (k a))) (Norm.norm (Inv.inv (k a)))
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem superpolynomialDecay_iff_isLittleO (hk : Tendsto k l atTop) :
    SuperpolynomialDecay l k f ↔ ∀ z : ℤ, f =o[l] fun a : α => k a ^ z := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    ⊢ Iff (Asymptotics.SuperpolynomialDecay l k f) (∀ (z : Int), Asymptotics.IsLit …
  -/
  refine ⟨fun h z => ?_, fun h => (superpolynomialDecay_iff_isBigO f hk).2 fun z => (h z).isBigO⟩
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : Asymptotics.SuperpolynomialDecay l k f
    z : Int
    ⊢ Asymptotics.IsLittleO l f fun a => HPow.hPow (k a) z
  -/
  have hk0 : ∀ᶠ x in l, k x ≠ 0 := hk.eventually_ne_atTop 0
  have : (fun _ : α => (1 : β)) =o[l] k :=
    isLittleO_of_tendsto' (hk0.mono fun x hkx hkx' => absurd hkx' hkx)
      (by simpa using hk.inv_tendsto_atTop)
  have : f =o[l] fun x : α => k x * k x ^ (z - 1) := by
    simpa using this.mul_isBigO ((superpolynomialDecay_iff_isBigO f hk).1 h <| z - 1)
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : Asymptotics.SuperpolynomialDecay l k f
    z : Int
    hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
    this✝ : Asymptotics.IsLittleO l (fun x => 1) k
    this : Asymptotics.IsLittleO l f fun x => HMul.hMul (k x) (HPow.hPow (k x) (HS …
    ⊢ Asymptotics.IsLittleO l f fun a => HPow.hPow (k a) z
  -/
  refine this.trans_isBigO (IsBigO.of_bound 1 (hk0.mono fun x hkx => le_of_eq ?_))
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    k f : α → β
    inst✝¹ : NormedLinearOrderedField β
    inst✝ : OrderTopology β
    hk : Filter.Tendsto k l Filter.atTop
    h : Asymptotics.SuperpolynomialDecay l k f
    z : Int
    hk0 : Filter.Eventually (fun x => Ne (k x) 0) l
    this✝ : Asymptotics.IsLittleO l (fun x => 1) k
    this : Asymptotics.IsLittleO l f fun x => HMul.hMul (k x) (HPow.hPow (k x) (HS …
    x : α
    hkx : Ne (k x) 0
    ⊢ Eq (Norm.norm (HMul.hMul (k x) (HPow.hPow (k x) (HSub.hSub z 1)))) (HMul.hMu …
  -/
  rw [one_mul, zpow_sub_one₀ hkx, mul_comm (k x), mul_assoc, inv_mul_cancel₀ hkx, mul_one]
  /-
    🎉 no goals
  -/


