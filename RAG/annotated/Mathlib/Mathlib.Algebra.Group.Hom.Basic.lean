/-- The `n`th power map on a commutative monoid for a natural `n`, considered as a morphism of
monoids. -/
@[to_additive (attr := simps) "Multiplication by a natural `n` on a commutative additive monoid,
considered as a morphism of additive monoids."]
def powMonoidHom (n : ℕ) : α →* α where
  toFun := (· ^ n)
  map_one' := one_pow _
  map_mul' a b := mul_pow a b n


/-- The `n`-th power map (for an integer `n`) on a commutative group, considered as a group
homomorphism. -/
@[to_additive (attr := simps) "Multiplication by an integer `n` on a commutative additive group,
considered as an additive group homomorphism."]
def zpowGroupHom (n : ℤ) : α →* α where
  toFun := (· ^ n)
  map_one' := one_zpow n
  map_mul' a b := mul_zpow a b n


/-- Inversion on a commutative group, considered as a monoid homomorphism. -/
@[to_additive "Negation on a commutative additive group, considered as an additive monoid
homomorphism."]
def invMonoidHom : α →* α where
  toFun := Inv.inv
  map_one' := inv_one
  map_mul' := mul_inv


@[simp]
theorem coe_invMonoidHom : (invMonoidHom : α → α) = Inv.inv := rfl


@[simp]
theorem invMonoidHom_apply (a : α) : invMonoidHom a = a⁻¹ := rfl


/-- Given two mul morphisms `f`, `g` to a commutative semigroup, `f * g` is the mul morphism
sending `x` to `f x * g x`. -/
@[to_additive "Given two additive morphisms `f`, `g` to an additive commutative semigroup,
`f + g` is the additive morphism sending `x` to `f x + g x`."]
instance [Mul M] [CommSemigroup N] : Mul (M →ₙ* N) :=
  ⟨fun f g =>
    { toFun := fun m => f m * g m,
      map_mul' := fun x y => by
        /-
          α : Type u_1
          M : Type u_2
          N : Type u_3
          P : Type u_4
          G : Type u_5
          H : Type u_6
          F : Type u_7
          inst✝¹ : Mul M
          inst✝ : CommSemigroup N
          f g : MulHom M N
          x y : M
          ⊢ Eq ((fun m => HMul.hMul (f m) (g m)) (HMul.hMul x y)) (HMul.hMul ((fun m =>  …
        -/
        show f (x * y) * g (x * y) = f x * g x * (f y * g y)
        /-
          α : Type u_1
          M : Type u_2
          N : Type u_3
          P : Type u_4
          G : Type u_5
          H : Type u_6
          F : Type u_7
          inst✝¹ : Mul M
          inst✝ : CommSemigroup N
          f g : MulHom M N
          x y : M
          ⊢ Eq (HMul.hMul (f (HMul.hMul x y)) (g (HMul.hMul x y))) (HMul.hMul (HMul.hMul …
        -/
        rw [f.map_mul, g.map_mul, ← mul_assoc, ← mul_assoc, mul_right_comm (f x)] }⟩
        /-
          🎉 no goals
        -/


@[to_additive (attr := simp)]
theorem mul_apply {M N} [Mul M] [CommSemigroup N] (f g : M →ₙ* N) (x : M) :
    (f * g) x = f x * g x := rfl


@[to_additive]
theorem mul_comp [Mul M] [Mul N] [CommSemigroup P] (g₁ g₂ : N →ₙ* P) (f : M →ₙ* N) :
    (g₁ * g₂).comp f = g₁.comp f * g₂.comp f := rfl


@[to_additive]
theorem comp_mul [Mul M] [CommSemigroup N] [CommSemigroup P] (g : N →ₙ* P) (f₁ f₂ : M →ₙ* N) :
    g.comp (f₁ * f₂) = g.comp f₁ * g.comp f₂ := by
  /-
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝² : Mul M
    inst✝¹ : CommSemigroup N
    inst✝ : CommSemigroup P
    g : MulHom N P
    f₁ f₂ : MulHom M N
    ⊢ Eq (g.comp (HMul.hMul f₁ f₂)) (HMul.hMul (g.comp f₁) (g.comp f₂))
  -/
  ext
  /-
    case h
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝² : Mul M
    inst✝¹ : CommSemigroup N
    inst✝ : CommSemigroup P
    g : MulHom N P
    f₁ f₂ : MulHom M N
    x✝ : M
    ⊢ Eq ((g.comp (HMul.hMul f₁ f₂)) x✝) ((HMul.hMul (g.comp f₁) (g.comp f₂)) x✝)
  -/
  simp only [mul_apply, Function.comp_apply, map_mul, coe_comp]
  /-
    🎉 no goals
  -/


/-- A homomorphism from a group to a monoid is injective iff its kernel is trivial.
For the iff statement on the triviality of the kernel, see `injective_iff_map_eq_one'`. -/
@[to_additive
  "A homomorphism from an additive group to an additive monoid is injective iff
  its kernel is trivial. For the iff statement on the triviality of the kernel,
  see `injective_iff_map_eq_zero'`."]
theorem _root_.injective_iff_map_eq_one {G H} [Group G] [MulOneClass H]
    [FunLike F G H] [MonoidHomClass F G H]
    (f : F) : Function.Injective f ↔ ∀ a, f a = 1 → a = 1 :=
  ⟨fun h _ => (map_eq_one_iff f h).mp, fun h x y hxy =>
                                  /-
                                    F : Type u_7
                                    G : Type u_8
                                    H : Type u_9
                                    inst✝³ : Group G
                                    inst✝² : MulOneClass H
                                    inst✝¹ : FunLike F G H
                                    inst✝ : MonoidHomClass F G H
                                    f : F
                                    h : ∀ (a : G), Eq (f a) 1 → Eq a 1
                                    x y : G
                                    hxy : Eq (f x) (f y)
                                    ⊢ Eq (f (HMul.hMul x (Inv.inv y))) 1
                                  -/
    mul_inv_eq_one.1 <| h _ <| by rw [map_mul, hxy, ← map_mul, mul_inv_cancel, map_one]⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- A homomorphism from a group to a monoid is injective iff its kernel is trivial,
stated as an iff on the triviality of the kernel.
For the implication, see `injective_iff_map_eq_one`. -/
@[to_additive
  "A homomorphism from an additive group to an additive monoid is injective iff its
  kernel is trivial, stated as an iff on the triviality of the kernel. For the implication, see
  `injective_iff_map_eq_zero`."]
theorem _root_.injective_iff_map_eq_one' {G H} [Group G] [MulOneClass H]
    [FunLike F G H] [MonoidHomClass F G H]
    (f : F) : Function.Injective f ↔ ∀ a, f a = 1 ↔ a = 1 :=
  (injective_iff_map_eq_one f).trans <|
    forall_congr' fun _ => ⟨fun h => ⟨h, fun H => H.symm ▸ map_one f⟩, Iff.mp⟩


/-- Makes a group homomorphism from a proof that the map preserves right division
`fun x y => x * y⁻¹`. See also `MonoidHom.of_map_div` for a version using `fun x y => x / y`.
-/
@[to_additive
  "Makes an additive group homomorphism from a proof that the map preserves
  the operation `fun a b => a + -b`. See also `AddMonoidHom.ofMapSub` for a version using
  `fun a b => a - b`."]
def ofMapMulInv {H : Type*} [Group H] (f : G → H)
    (map_div : ∀ a b : G, f (a * b⁻¹) = f a * (f b)⁻¹) : G →* H :=
  (mk' f) fun x y =>
    calc
      f (x * y) = f x * (f <| 1 * 1⁻¹ * y⁻¹)⁻¹ := by
        /-
          α : Type u_1
          M : Type u_2
          N : Type u_3
          P : Type u_4
          G : Type u_5
          H✝ : Type u_6
          F : Type u_7
          inst✝¹ : Group G
          H : Type u_8
          inst✝ : Group H
          f : G → H
          map_div : ∀ (a b : G), Eq (f (HMul.hMul a (Inv.inv b))) (HMul.hMul (f a) (Inv. …
          x y : G
          ⊢ Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (Inv.inv (f (HMul.hMul (HMul.hMul 1  …
        -/
        { simp only [one_mul, inv_one, ← map_div, inv_inv] }
        /-
          🎉 no goals
        -/
      _ = f x * f y := by
        { simp only [map_div]
          simp only [mul_inv_cancel, one_mul, inv_inv] }


@[to_additive (attr := simp)]
theorem coe_of_map_mul_inv {H : Type*} [Group H] (f : G → H)
    (map_div : ∀ a b : G, f (a * b⁻¹) = f a * (f b)⁻¹) :
  ↑(ofMapMulInv f map_div) = f := rfl


/-- Define a morphism of additive groups given a map which respects ratios. -/
@[to_additive "Define a morphism of additive groups given a map which respects difference."]
def ofMapDiv {H : Type*} [Group H] (f : G → H) (hf : ∀ x y, f (x / y) = f x / f y) : G →* H :=
                    /-
                      α : Type u_1
                      M : Type u_2
                      N : Type u_3
                      P : Type u_4
                      G : Type u_5
                      H✝ : Type u_6
                      F : Type u_7
                      inst✝¹ : Group G
                      H : Type u_8
                      inst✝ : Group H
                      f : G → H
                      hf : ∀ (x y : G), Eq (f (HDiv.hDiv x y)) (HDiv.hDiv (f x) (f y))
                      ⊢ ∀ (a b : G), Eq (f (HMul.hMul a (Inv.inv b))) (HMul.hMul (f a) (Inv.inv (f b …
                    -/
  ofMapMulInv f (by simpa only [div_eq_mul_inv] using hf)
                    /-
                      🎉 no goals
                    -/


@[to_additive (attr := simp)]
theorem coe_of_map_div {H : Type*} [Group H] (f : G → H) (hf : ∀ x y, f (x / y) = f x / f y) :
    ↑(ofMapDiv f hf) = f := rfl


/-- Given two monoid morphisms `f`, `g` to a commutative monoid, `f * g` is the monoid morphism
sending `x` to `f x * g x`. -/
@[to_additive]
instance mul : Mul (M →* N) :=
  ⟨fun f g =>
    { toFun := fun m => f m * g m,
                                        /-
                                          α : Type u_1
                                          M : Type u_2
                                          N : Type u_3
                                          P : Type u_4
                                          G : Type u_5
                                          H : Type u_6
                                          F : Type u_7
                                          inst✝¹ : MulOneClass M
                                          inst✝ : CommMonoid N
                                          f g : MonoidHom M N
                                          ⊢ Eq (HMul.hMul (f 1) (g 1)) 1
                                        -/
      map_one' := show f 1 * g 1 = 1 by simp,
                                        /-
                                          🎉 no goals
                                        -/
      map_mul' := fun x y => by
        /-
          α : Type u_1
          M : Type u_2
          N : Type u_3
          P : Type u_4
          G : Type u_5
          H : Type u_6
          F : Type u_7
          inst✝¹ : MulOneClass M
          inst✝ : CommMonoid N
          f g : MonoidHom M N
          x y : M
          ⊢ Eq ({ toFun := fun m => HMul.hMul (f m) (g m), map_one' := ⋯ }.toFun (HMul.h …
        -/
        show f (x * y) * g (x * y) = f x * g x * (f y * g y)
        /-
          α : Type u_1
          M : Type u_2
          N : Type u_3
          P : Type u_4
          G : Type u_5
          H : Type u_6
          F : Type u_7
          inst✝¹ : MulOneClass M
          inst✝ : CommMonoid N
          f g : MonoidHom M N
          x y : M
          ⊢ Eq (HMul.hMul (f (HMul.hMul x y)) (g (HMul.hMul x y))) (HMul.hMul (HMul.hMul …
        -/
        rw [f.map_mul, g.map_mul, ← mul_assoc, ← mul_assoc, mul_right_comm (f x)] }⟩
        /-
          🎉 no goals
        -/


@[to_additive (attr := simp)] lemma mul_apply (f g : M →* N) (x : M) : (f * g) x = f x * g x := rfl


@[to_additive]
lemma mul_comp [MulOneClass P] (g₁ g₂ : M →* N) (f : P →* M) :
    (g₁ * g₂).comp f = g₁.comp f * g₂.comp f := rfl


@[to_additive]
lemma comp_mul [CommMonoid P] (g : N →* P) (f₁ f₂ : M →* N) :
    g.comp (f₁ * f₂) = g.comp f₁ * g.comp f₂ := by
  /-
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝² : MulOneClass M
    inst✝¹ : CommMonoid N
    inst✝ : CommMonoid P
    g : MonoidHom N P
    f₁ f₂ : MonoidHom M N
    ⊢ Eq (g.comp (HMul.hMul f₁ f₂)) (HMul.hMul (g.comp f₁) (g.comp f₂))
  -/
  ext; simp only [mul_apply, Function.comp_apply, map_mul, coe_comp]
       /-
         🎉 no goals
       -/


/-- If `f` is a monoid homomorphism to a commutative group, then `f⁻¹` is the homomorphism sending
`x` to `(f x)⁻¹`. -/
@[to_additive "If `f` is an additive monoid homomorphism to an additive commutative group,
then `-f` is the homomorphism sending `x` to `-(f x)`."]
instance : Inv (M →* G) where
                                              /-
                                                α : Type u_1
                                                M : Type u_2
                                                N : Type u_3
                                                P : Type u_4
                                                G : Type u_5
                                                H : Type u_6
                                                F : Type u_7
                                                inst✝³ : MulOneClass M
                                                inst✝² : MulOneClass N
                                                inst✝¹ : CommGroup G
                                                inst✝ : CommGroup H
                                                f : MonoidHom M G
                                                a b : M
                                                ⊢ Eq ((fun g => Inv.inv (f g)) (HMul.hMul a b)) (HMul.hMul ((fun g => Inv.inv  …
                                              -/
  inv f := mk' (fun g ↦ (f g)⁻¹) fun a b ↦ by simp_rw [← mul_inv, f.map_mul]
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive (attr := simp)] lemma inv_apply (f : M →* G) (x : M) : f⁻¹ x = (f x)⁻¹ := rfl


@[to_additive (attr := simp)]
theorem inv_comp (φ : N →* G) (ψ : M →* N) : φ⁻¹.comp ψ = (φ.comp ψ)⁻¹ := rfl


@[to_additive (attr := simp)]
theorem comp_inv (φ : G →* H) (ψ : M →* G) : φ.comp ψ⁻¹ = (φ.comp ψ)⁻¹ := by
  /-
    M : Type u_2
    G : Type u_5
    H : Type u_6
    inst✝² : MulOneClass M
    inst✝¹ : CommGroup G
    inst✝ : CommGroup H
    φ : MonoidHom G H
    ψ : MonoidHom M G
    ⊢ Eq (φ.comp (Inv.inv ψ)) (Inv.inv (φ.comp ψ))
  -/
  ext
  /-
    case h
    M : Type u_2
    G : Type u_5
    H : Type u_6
    inst✝² : MulOneClass M
    inst✝¹ : CommGroup G
    inst✝ : CommGroup H
    φ : MonoidHom G H
    ψ : MonoidHom M G
    x✝ : M
    ⊢ Eq ((φ.comp (Inv.inv ψ)) x✝) ((Inv.inv (φ.comp ψ)) x✝)
  -/
  simp only [Function.comp_apply, inv_apply, map_inv, coe_comp]
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are monoid homomorphisms to a commutative group, then `f / g` is the homomorphism
sending `x` to `(f x) / (g x)`. -/
@[to_additive "If `f` and `g` are monoid homomorphisms to an additive commutative group,
then `f - g` is the homomorphism sending `x` to `(f x) - (g x)`."]
instance : Div (M →* G) where
  div f g := mk' (fun x ↦ f x / g x) fun a b ↦ by
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      G : Type u_5
      H : Type u_6
      F : Type u_7
      inst✝³ : MulOneClass M
      inst✝² : MulOneClass N
      inst✝¹ : CommGroup G
      inst✝ : CommGroup H
      f g : MonoidHom M G
      a b : M
      ⊢ Eq ((fun x => HDiv.hDiv (f x) (g x)) (HMul.hMul a b)) (HMul.hMul ((fun x =>  …
    -/
    simp [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)] lemma div_apply (f g : M →* G) (x : M) : (f / g) x = f x / g x := rfl


@[to_additive (attr := simp)]
lemma div_comp (f g : N →* G) (h : M →* N) : (f / g).comp h = f.comp h / g.comp h := rfl


@[to_additive (attr := simp)]
lemma comp_div (f : G →* H) (g h : M →* G) : f.comp (g / h) = f.comp g / f.comp h := by
  /-
    M : Type u_2
    G : Type u_5
    H : Type u_6
    inst✝² : MulOneClass M
    inst✝¹ : CommGroup G
    inst✝ : CommGroup H
    f : MonoidHom G H
    g h : MonoidHom M G
    ⊢ Eq (f.comp (HDiv.hDiv g h)) (HDiv.hDiv (f.comp g) (f.comp h))
  -/
  ext; simp only [Function.comp_apply, div_apply, map_div, coe_comp]
       /-
         🎉 no goals
       -/


/-- If `H` is commutative and `G →* H` is injective, then `G` is commutative. -/
def commGroupOfInjective [Group G] [CommGroup H] (f : G →* H) (hf : Function.Injective f) :
    CommGroup G :=
      /-
        α : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        G : Type u_5
        H : Type u_6
        F : Type u_7
        inst✝¹ : Group G
        inst✝ : CommGroup H
        f : MonoidHom G H
        hf : Function.Injective ⇑f
        ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
  ⟨by simp_rw [← hf.eq_iff, map_mul, mul_comm, implies_true]⟩
      /-
        🎉 no goals
      -/


/-- If `G` is commutative and `G →* H` is surjective, then `H` is commutative. -/
def commGroupOfSurjective [CommGroup G] [Group H] (f : G →* H) (hf : Function.Surjective f) :
    CommGroup H :=
      /-
        α : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        G : Type u_5
        H : Type u_6
        F : Type u_7
        inst✝¹ : CommGroup G
        inst✝ : Group H
        f : MonoidHom G H
        hf : Function.Surjective ⇑f
        ⊢ ∀ (a b : H), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
  ⟨by simp_rw [hf.forall₂, ← map_mul, mul_comm, implies_true]⟩
      /-
        🎉 no goals
      -/


