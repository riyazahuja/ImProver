/-- If `f : α →+* β` is bounded with respect to a ring seminorm `nα` on `α` and a
  power-multiplicative function `nβ : β → ℝ`, then `∀ x : α, nβ (f x) ≤ nα x`. -/
theorem contraction_of_isPowMul_of_boundedWrt {F : Type*} {α : outParam (Type*)} [Ring α]
    [FunLike F α ℝ] [RingSeminormClass F α ℝ] {β : Type*} [Ring β] (nα : F) {nβ : β → ℝ}
    (hβ : IsPowMul nβ) {f : α →+* β} (hf : f.IsBoundedWrt nα nβ) (x : α) : nβ (f x) ≤ nα x := by
  /-
    F : Type u_1
    α : outParam (Type u_2)
    inst✝³ : Ring α
    inst✝² : FunLike F α Real
    inst✝¹ : RingSeminormClass F α Real
    β : Type u_3
    inst✝ : Ring β
    nα : F
    nβ : β → Real
    hβ : IsPowMul nβ
    f : RingHom α β
    hf : RingHom.IsBoundedWrt (⇑nα) nβ f
    x : α
    ⊢ LE.le (nβ (f x)) (nα x)
  -/
  obtain ⟨C, hC0, hC⟩ := hf
  have hlim : Tendsto (fun n : ℕ => C ^ (1 / (n : ℝ)) * nα x) atTop (𝓝 (nα x)) := by
    nth_rewrite 2 [← one_mul (nα x)]
    exact ((rpow_zero C ▸ ContinuousAt.tendsto (continuousAt_const_rpow (ne_of_gt hC0))).comp
      (tendsto_const_div_atTop_nhds_zero_nat 1)).mul tendsto_const_nhds
  /-
    case intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝³ : Ring α
    inst✝² : FunLike F α Real
    inst✝¹ : RingSeminormClass F α Real
    β : Type u_3
    inst✝ : Ring β
    nα : F
    nβ : β → Real
    hβ : IsPowMul nβ
    f : RingHom α β
    x : α
    C : Real
    hC0 : LT.lt 0 C
    hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
    hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
    ⊢ LE.le (nβ (f x)) (nα x)
  -/
  apply ge_of_tendsto hlim
  /-
    case intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝³ : Ring α
    inst✝² : FunLike F α Real
    inst✝¹ : RingSeminormClass F α Real
    β : Type u_3
    inst✝ : Ring β
    nα : F
    nβ : β → Real
    hβ : IsPowMul nβ
    f : RingHom α β
    x : α
    C : Real
    hC0 : LT.lt 0 C
    hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
    hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
    ⊢ Filter.Eventually (fun c => LE.le (nβ (f x)) (HMul.hMul (HPow.hPow C (HDiv.h …
  -/
  simp only [eventually_atTop, ge_iff_le]
  /-
    case intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝³ : Ring α
    inst✝² : FunLike F α Real
    inst✝¹ : RingSeminormClass F α Real
    β : Type u_3
    inst✝ : Ring β
    nα : F
    nβ : β → Real
    hβ : IsPowMul nβ
    f : RingHom α β
    x : α
    C : Real
    hC0 : LT.lt 0 C
    hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
    hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
    ⊢ Exists fun a => ∀ (b : Nat), LE.le a b → LE.le (nβ (f x)) (HMul.hMul (HPow.h …
  -/
  use 1
  /-
    case h
    F : Type u_1
    α : outParam (Type u_2)
    inst✝³ : Ring α
    inst✝² : FunLike F α Real
    inst✝¹ : RingSeminormClass F α Real
    β : Type u_3
    inst✝ : Ring β
    nα : F
    nβ : β → Real
    hβ : IsPowMul nβ
    f : RingHom α β
    x : α
    C : Real
    hC0 : LT.lt 0 C
    hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
    hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
    ⊢ ∀ (b : Nat), LE.le 1 b → LE.le (nβ (f x)) (HMul.hMul (HPow.hPow C (HDiv.hDiv …
  -/
  intro n hn
  have h : (C ^ (1 / n : ℝ)) ^ n = C := by
    have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hn)
    rw [← rpow_natCast, ← rpow_mul (le_of_lt hC0), one_div, inv_mul_cancel₀ hn0, rpow_one]
  apply le_of_pow_le_pow_left₀ (ne_of_gt hn)
    (mul_nonneg (rpow_nonneg (le_of_lt hC0) _) (apply_nonneg _ _))
    /-
      case h
      F : Type u_1
      α : outParam (Type u_2)
      inst✝³ : Ring α
      inst✝² : FunLike F α Real
      inst✝¹ : RingSeminormClass F α Real
      β : Type u_3
      inst✝ : Ring β
      nα : F
      nβ : β → Real
      hβ : IsPowMul nβ
      f : RingHom α β
      x : α
      C : Real
      hC0 : LT.lt 0 C
      hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
      hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
      n : Nat
      hn : LE.le 1 n
      h : Eq (HPow.hPow (HPow.hPow C (HDiv.hDiv 1 ↑n)) n) C
      ⊢ LE.le (HPow.hPow (nβ (f x)) n) (HPow.hPow (HMul.hMul (HPow.hPow C (HDiv.hDiv …
    -/
  · rw [mul_pow, h, ← hβ _ hn, ← RingHom.map_pow]
    /-
      case h
      F : Type u_1
      α : outParam (Type u_2)
      inst✝³ : Ring α
      inst✝² : FunLike F α Real
      inst✝¹ : RingSeminormClass F α Real
      β : Type u_3
      inst✝ : Ring β
      nα : F
      nβ : β → Real
      hβ : IsPowMul nβ
      f : RingHom α β
      x : α
      C : Real
      hC0 : LT.lt 0 C
      hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
      hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
      n : Nat
      hn : LE.le 1 n
      h : Eq (HPow.hPow (HPow.hPow C (HDiv.hDiv 1 ↑n)) n) C
      ⊢ LE.le (nβ (f (HPow.hPow x n))) (HMul.hMul C (HPow.hPow (nα x) n))
    -/
    apply le_trans (hC (x ^ n))
    /-
      case h
      F : Type u_1
      α : outParam (Type u_2)
      inst✝³ : Ring α
      inst✝² : FunLike F α Real
      inst✝¹ : RingSeminormClass F α Real
      β : Type u_3
      inst✝ : Ring β
      nα : F
      nβ : β → Real
      hβ : IsPowMul nβ
      f : RingHom α β
      x : α
      C : Real
      hC0 : LT.lt 0 C
      hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
      hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
      n : Nat
      hn : LE.le 1 n
      h : Eq (HPow.hPow (HPow.hPow C (HDiv.hDiv 1 ↑n)) n) C
      ⊢ LE.le (HMul.hMul C (nα (HPow.hPow x n))) (HMul.hMul C (HPow.hPow (nα x) n))
    -/
    rw [mul_le_mul_left hC0]
    /-
      case h
      F : Type u_1
      α : outParam (Type u_2)
      inst✝³ : Ring α
      inst✝² : FunLike F α Real
      inst✝¹ : RingSeminormClass F α Real
      β : Type u_3
      inst✝ : Ring β
      nα : F
      nβ : β → Real
      hβ : IsPowMul nβ
      f : RingHom α β
      x : α
      C : Real
      hC0 : LT.lt 0 C
      hC : ∀ (x : α), LE.le (nβ (f x)) (HMul.hMul C (nα x))
      hlim : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow C (HDiv.hDiv 1 ↑n)) (nα x …
      n : Nat
      hn : LE.le 1 n
      h : Eq (HPow.hPow (HPow.hPow C (HDiv.hDiv 1 ↑n)) n) C
      ⊢ LE.le (nα (HPow.hPow x n)) (HPow.hPow (nα x) n)
    -/
    exact map_pow_le_pow _ _ (Nat.one_le_iff_ne_zero.mp hn)
    /-
      🎉 no goals
    -/


/-- Given a bounded `f : α →+* β` between seminormed rings, is the seminorm on `β` is
  power-multiplicative, then `f` is a contraction. -/
theorem contraction_of_isPowMul {α β : Type*} [SeminormedRing α] [SeminormedRing β]
    (hβ : IsPowMul (norm : β → ℝ)) {f : α →+* β} (hf : f.IsBounded) (x : α) : norm (f x) ≤ norm x :=
  contraction_of_isPowMul_of_boundedWrt (SeminormedRing.toRingSeminorm α) hβ hf x


/-- Given two power-multiplicative ring seminorms `f, g` on `α`, if `f` is bounded by a positive
  multiple of `g` and vice versa, then `f = g`. -/
theorem eq_seminorms {F : Type*} {α : outParam (Type*)} [Ring α] [FunLike F α ℝ]
    [RingSeminormClass F α ℝ] {f g : F} (hfpm : IsPowMul f) (hgpm : IsPowMul g)
    (hfg : ∃ (r : ℝ) (_ : 0 < r), ∀ a : α, f a ≤ r * g a)
    (hgf : ∃ (r : ℝ) (_ : 0 < r), ∀ a : α, g a ≤ r * f a) : f = g := by
  /-
    F : Type u_1
    α : outParam (Type u_2)
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f g : F
    hfpm : IsPowMul ⇑f
    hgpm : IsPowMul ⇑g
    hfg : Exists fun r => Exists fun x => ∀ (a : α), LE.le (f a) (HMul.hMul r (g a))
    hgf : Exists fun r => Exists fun x => ∀ (a : α), LE.le (g a) (HMul.hMul r (f a))
    ⊢ Eq f g
  -/
  obtain ⟨r, hr0, hr⟩ := hfg
  /-
    case intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f g : F
    hfpm : IsPowMul ⇑f
    hgpm : IsPowMul ⇑g
    hgf : Exists fun r => Exists fun x => ∀ (a : α), LE.le (g a) (HMul.hMul r (f a))
    r : Real
    hr0 : LT.lt 0 r
    hr : ∀ (a : α), LE.le (f a) (HMul.hMul r (g a))
    ⊢ Eq f g
  -/
  obtain ⟨s, hs0, hs⟩ := hgf
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f g : F
    hfpm : IsPowMul ⇑f
    hgpm : IsPowMul ⇑g
    r : Real
    hr0 : LT.lt 0 r
    hr : ∀ (a : α), LE.le (f a) (HMul.hMul r (g a))
    s : Real
    hs0 : LT.lt 0 s
    hs : ∀ (a : α), LE.le (g a) (HMul.hMul s (f a))
    ⊢ Eq f g
  -/
  have hle : RingHom.IsBoundedWrt f g (RingHom.id _) := ⟨s, hs0, hs⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f g : F
    hfpm : IsPowMul ⇑f
    hgpm : IsPowMul ⇑g
    r : Real
    hr0 : LT.lt 0 r
    hr : ∀ (a : α), LE.le (f a) (HMul.hMul r (g a))
    s : Real
    hs0 : LT.lt 0 s
    hs : ∀ (a : α), LE.le (g a) (HMul.hMul s (f a))
    hle : RingHom.IsBoundedWrt (⇑f) (⇑g) (RingHom.id α)
    ⊢ Eq f g
  -/
  have hge : RingHom.IsBoundedWrt g f (RingHom.id _) := ⟨r, hr0, hr⟩
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f g : F
    hfpm : IsPowMul ⇑f
    hgpm : IsPowMul ⇑g
    r : Real
    hr0 : LT.lt 0 r
    hr : ∀ (a : α), LE.le (f a) (HMul.hMul r (g a))
    s : Real
    hs0 : LT.lt 0 s
    hs : ∀ (a : α), LE.le (g a) (HMul.hMul s (f a))
    hle : RingHom.IsBoundedWrt (⇑f) (⇑g) (RingHom.id α)
    hge : RingHom.IsBoundedWrt (⇑g) (⇑f) (RingHom.id α)
    ⊢ Eq f g
  -/
  rw [← Function.Injective.eq_iff DFunLike.coe_injective']
  /-
    case intro.intro.intro.intro
    F : Type u_1
    α : outParam (Type u_2)
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f g : F
    hfpm : IsPowMul ⇑f
    hgpm : IsPowMul ⇑g
    r : Real
    hr0 : LT.lt 0 r
    hr : ∀ (a : α), LE.le (f a) (HMul.hMul r (g a))
    s : Real
    hs0 : LT.lt 0 s
    hs : ∀ (a : α), LE.le (g a) (HMul.hMul s (f a))
    hle : RingHom.IsBoundedWrt (⇑f) (⇑g) (RingHom.id α)
    hge : RingHom.IsBoundedWrt (⇑g) (⇑f) (RingHom.id α)
    ⊢ Eq ⇑f ⇑g
  -/
  ext x
  exact le_antisymm (contraction_of_isPowMul_of_boundedWrt g hfpm hge x)
    (contraction_of_isPowMul_of_boundedWrt f hgpm hle x)


/-- If `R` is a normed commutative ring and `f₁` and `f₂` are two power-multiplicative `R`-algebra
  norms on `S`, then if `f₁` and `f₂` are equivalent on every  subring `R[y]` for `y : S`, it
  follows that `f₁ = f₂` [BGR, Proposition 3.1.5/1][bosch-guntzer-remmert]. -/
theorem eq_of_powMul_faithful (f₁ : AlgebraNorm R S) (hf₁_pm : IsPowMul f₁) (f₂ : AlgebraNorm R S)
    (hf₂_pm : IsPowMul f₂)
    (h_eq : ∀ y : S, ∃ (C₁ C₂ : ℝ) (_ : 0 < C₁) (_ : 0 < C₂),
      ∀ x : Algebra.adjoin R {y}, f₁ x.val ≤ C₁ * f₂ x.val ∧ f₂ x.val ≤ C₂ * f₁ x.val) :
    f₁ = f₂ := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    ⊢ Eq f₁ f₂
  -/
  ext x
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  set g₁ : AlgebraNorm R (Algebra.adjoin R ({x} : Set S)) := AlgebraNorm.restriction _ f₁
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  set g₂ : AlgebraNorm R (Algebra.adjoin R ({x} : Set S)) := AlgebraNorm.restriction _ f₂
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  have hg₁_pm : IsPowMul g₁ := IsPowMul.restriction _ hf₁_pm
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  have hg₂_pm : IsPowMul g₂ := IsPowMul.restriction _ hf₂_pm
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  let y : Algebra.adjoin R ({x} : Set S) := ⟨x, Algebra.self_mem_adjoin_singleton R x⟩
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  have hy : x = y.val := rfl
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    hy : Eq x ↑y
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  have h1 : f₁ y.val = g₁ y := rfl
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    hy : Eq x ↑y
    h1 : Eq (f₁ ↑y) (g₁ y)
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  have h2 : f₂ y.val = g₂ y := rfl
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    hy : Eq x ↑y
    h1 : Eq (f₁ ↑y) (g₁ y)
    h2 : Eq (f₂ ↑y) (g₂ y)
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  obtain ⟨C₁, C₂, hC₁_pos, hC₂_pos, hC⟩ := h_eq x
  /-
    case a.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    hy : Eq x ↑y
    h1 : Eq (f₁ ↑y) (g₁ y)
    h2 : Eq (f₂ ↑y) (g₂ y)
    C₁ C₂ : Real
    hC₁_pos : LT.lt 0 C₁
    hC₂_pos : LT.lt 0 C₂
    hC : ∀ (x_1 : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.s …
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  obtain ⟨hC₁, hC₂⟩ := forall_and.mp hC
  /-
    case a.intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f₁ : AlgebraNorm R S
    hf₁_pm : IsPowMul ⇑f₁
    f₂ : AlgebraNorm R S
    hf₂_pm : IsPowMul ⇑f₂
    h_eq : ∀ (y : S), Exists fun C₁ => Exists fun C₂ => Exists fun x => Exists fun …
    x : S
    g₁ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    g₂ : AlgebraNorm R (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singl …
    hg₁_pm : IsPowMul ⇑g₁
    hg₂_pm : IsPowMul ⇑g₂
    y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
    hy : Eq x ↑y
    h1 : Eq (f₁ ↑y) (g₁ y)
    h2 : Eq (f₂ ↑y) (g₂ y)
    C₁ C₂ : Real
    hC₁_pos : LT.lt 0 C₁
    hC₂_pos : LT.lt 0 C₂
    hC : ∀ (x_1 : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.s …
    hC₁ : ∀ (x_1 : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton. …
    hC₂ : ∀ (x_1 : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton. …
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  rw [hy, h1, h2, eq_seminorms hg₁_pm hg₂_pm ⟨C₁, hC₁_pos, hC₁⟩ ⟨C₂, hC₂_pos, hC₂⟩]
  /-
    🎉 no goals
  -/

