@[to_additive] instance : One (DirectLimit G f) where
  one := map₀ f fun _ ↦ 1


@[to_additive] theorem one_def (i) : (1 : DirectLimit G f) = ⟦⟨i, 1⟩⟧ :=
  map₀_def _ _ (fun _ _ _ ↦ map_one _) _


@[to_additive] theorem exists_eq_one (x) :
    ⟦x⟧ = (1 : DirectLimit G f) ↔ ∃ i h, f x.1 i h x.2 = 1 := by
  /-
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    T : ⦃i j : ι⦄ → LE.le i j → Type u_4
    f : (x x_1 : ι) → (h : LE.le x x_1) → T h
    inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
    inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : Nonempty ι
    inst✝¹ : (i : ι) → One (G i)
    inst✝ : ∀ (i j : ι) (h : LE.le i j), OneHomClass (T h) (G i) (G j)
    x : Sigma fun i => G i
    ⊢ Iff (Eq (Quotient.mk (DirectLimit.setoid f) x) 1) (Exists fun i => Exists fu …
  -/
  rw [one_def x.1, Quotient.eq]
  exact ⟨fun ⟨i, h, _, eq⟩ ↦ ⟨i, h, eq.trans (map_one _)⟩,
    fun ⟨i, h, eq⟩ ↦ ⟨i, h, h, eq.trans (map_one _).symm⟩⟩


@[to_additive] instance : Mul (DirectLimit G f) where
  mul := map₂ f f f (fun _ ↦ (· * ·)) fun _ _ _ ↦ map_mul _


@[to_additive] theorem mul_def (i) (x y : G i) :
    ⟦⟨i, x⟩⟧ * ⟦⟨i, y⟩⟧ = (⟦⟨i, x * y⟩⟧ : DirectLimit G f) :=
  map₂_def ..


@[to_additive] instance [∀ i, CommMagma (G i)] [∀ i j h, MulHomClass (T h) (G i) (G j)] :
    CommMagma (DirectLimit G f) where
                                                      /-
                                                        R : Type u_1
                                                        ι : Type u_2
                                                        inst✝⁵ : Preorder ι
                                                        G : ι → Type u_3
                                                        T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                        f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                        inst✝⁴ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                        inst✝³ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                        inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                        inst✝¹ : (i : ι) → CommMagma (G i)
                                                        inst✝ : ∀ (i j : ι) (h : LE.le i j), MulHomClass (T h) (G i) (G j)
                                                        i : ι
                                                        x✝¹ x✝ : G i
                                                        ⊢ Eq (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝¹⟩) (Quotient.mk (Di …
                                                      -/
  mul_comm := DirectLimit.induction₂ _ fun i _ _ ↦ by simp_rw [mul_def, mul_comm]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive] instance [∀ i, Semigroup (G i)] [∀ i j h, MulHomClass (T h) (G i) (G j)] :
    Semigroup (DirectLimit G f) where
                                                         /-
                                                           R : Type u_1
                                                           ι : Type u_2
                                                           inst✝⁵ : Preorder ι
                                                           G : ι → Type u_3
                                                           T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                           f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                           inst✝⁴ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                           inst✝³ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                           inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                           inst✝¹ : (i : ι) → Semigroup (G i)
                                                           inst✝ : ∀ (i j : ι) (h : LE.le i j), MulHomClass (T h) (G i) (G j)
                                                           i : ι
                                                           x✝² x✝¹ x✝ : G i
                                                           ⊢ Eq (HMul.hMul (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝²⟩) (Quot …
                                                         -/
  mul_assoc := DirectLimit.induction₃ _ fun i _ _ _ ↦ by simp_rw [mul_def, mul_assoc]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive] instance [∀ i, CommSemigroup (G i)] [∀ i j h, MulHomClass (T h) (G i) (G j)] :
    CommSemigroup (DirectLimit G f) where
  mul_comm := mul_comm


@[to_additive] instance : SMul R (DirectLimit G f) where
  smul r := map _ _ (fun _ ↦ (r • ·)) fun _ _ _ ↦ map_smul _ r


@[to_additive] theorem smul_def (i x) (r : R) : r • ⟦⟨i, x⟩⟧ = (⟦⟨i, r • x⟩⟧ : DirectLimit G f) :=
  rfl


@[to_additive] instance [Monoid R] [∀ i, MulAction R (G i)]
    [∀ i j h, MulActionHomClass (T h) R (G i) (G j)] :
    MulAction R (DirectLimit G f) where
                                                   /-
                                                     R : Type u_1
                                                     ι : Type u_2
                                                     inst✝⁶ : Preorder ι
                                                     G : ι → Type u_3
                                                     T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                     f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                     inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                     inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                     inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                     inst✝² : Monoid R
                                                     inst✝¹ : (i : ι) → MulAction R (G i)
                                                     inst✝ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
                                                     i : ι
                                                     x✝ : G i
                                                     ⊢ Eq (HSMul.hSMul 1 (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩)) (Quotient.mk …
                                                   -/
  one_smul := DirectLimit.induction _ fun i _ ↦ by rw [smul_def, one_smul]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                       /-
                                                         R : Type u_1
                                                         ι : Type u_2
                                                         inst✝⁶ : Preorder ι
                                                         G : ι → Type u_3
                                                         T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                         f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                         inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                         inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                         inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                         inst✝² : Monoid R
                                                         inst✝¹ : (i : ι) → MulAction R (G i)
                                                         inst✝ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
                                                         x✝² x✝¹ : R
                                                         i : ι
                                                         x✝ : G i
                                                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) (Quotient.mk (DirectLimit.setoid f) ⟨i,  …
                                                       -/
  mul_smul _ _ := DirectLimit.induction _ fun i _ ↦ by simp_rw [smul_def, mul_smul]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive] instance [∀ i, MulOneClass (G i)] [∀ i j h, MonoidHomClass (T h) (G i) (G j)] :
    MulOneClass (DirectLimit G f) where
                                                  /-
                                                    R : Type u_1
                                                    ι : Type u_2
                                                    inst✝⁶ : Preorder ι
                                                    G : ι → Type u_3
                                                    T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                    f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                    inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                    inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                    inst✝² : Nonempty ι
                                                    inst✝¹ : (i : ι) → MulOneClass (G i)
                                                    inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
                                                    i : ι
                                                    x✝ : G i
                                                    ⊢ Eq (HMul.hMul 1 (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩)) (Quotient.mk ( …
                                                  -/
  one_mul := DirectLimit.induction _ fun i _ ↦ by simp_rw [one_def i, mul_def, one_mul]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    R : Type u_1
                                                    ι : Type u_2
                                                    inst✝⁶ : Preorder ι
                                                    G : ι → Type u_3
                                                    T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                    f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                    inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                    inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                    inst✝² : Nonempty ι
                                                    inst✝¹ : (i : ι) → MulOneClass (G i)
                                                    inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
                                                    i : ι
                                                    x✝ : G i
                                                    ⊢ Eq (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩) 1) (Quotient.mk ( …
                                                  -/
  mul_one := DirectLimit.induction _ fun i _ ↦ by simp_rw [one_def i, mul_def, mul_one]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive] instance : Monoid (DirectLimit G f) where
  one_mul := one_mul
  mul_one := mul_one
  npow n := map _ _ (fun _ ↦ (· ^ n)) fun _ _ _ x ↦ map_pow _ x n
                                                    /-
                                                      R : Type u_1
                                                      ι : Type u_2
                                                      inst✝⁶ : Preorder ι
                                                      G : ι → Type u_3
                                                      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                      inst✝² : Nonempty ι
                                                      inst✝¹ : (i : ι) → Monoid (G i)
                                                      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
                                                      i : ι
                                                      x✝ : G i
                                                      ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) 0 (Quoti …
                                                    -/
  npow_zero := DirectLimit.induction _ fun i _ ↦ by simp_rw [map_def, pow_zero, one_def i]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                      /-
                                                        R : Type u_1
                                                        ι : Type u_2
                                                        inst✝⁶ : Preorder ι
                                                        G : ι → Type u_3
                                                        T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                        f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                        inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                        inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                        inst✝² : Nonempty ι
                                                        inst✝¹ : (i : ι) → Monoid (G i)
                                                        inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
                                                        n : Nat
                                                        i : ι
                                                        x✝ : G i
                                                        ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) (HAdd.hA …
                                                      -/
  npow_succ n := DirectLimit.induction _ fun i _ ↦ by simp_rw [map_def, pow_succ, mul_def]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive] theorem npow_def (i x) (n : ℕ) : ⟦⟨i, x⟩⟧ ^ n = (⟦⟨i, x ^ n⟩⟧ : DirectLimit G f) :=
  rfl


@[to_additive] instance [∀ i, CommMonoid (G i)] [∀ i j h, MonoidHomClass (T h) (G i) (G j)] :
    CommMonoid (DirectLimit G f) where
  mul_comm := mul_comm


@[to_additive] instance : Group (DirectLimit G f) where
  inv := map _ _ (fun _ ↦ (·⁻¹)) fun _ _ _ ↦ map_inv _
  div := map₂ _ _ _ (fun _ ↦ (· / ·)) fun _ _ _ ↦ map_div _
  zpow n := map _ _ (fun _ ↦ (· ^ n)) fun _ _ _ x ↦ map_zpow _ x n
  div_eq_mul_inv := DirectLimit.induction₂ _ fun i _ _ ↦ show map₂ .. = _ * map .. by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → Group (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
      i : ι
      x✝¹ x✝ : G i
      ⊢ Eq (DirectLimit.map₂ f f f (fun x x1 x2 => HDiv.hDiv x1 x2) ⋯ (Quotient.mk ( …
    -/
    simp_rw [map₂_def, map_def, div_eq_mul_inv, mul_def]
    /-
      🎉 no goals
    -/
                                                     /-
                                                       R : Type u_1
                                                       ι : Type u_2
                                                       inst✝⁶ : Preorder ι
                                                       G : ι → Type u_3
                                                       T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                       f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                       inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                       inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                       inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                       inst✝² : Nonempty ι
                                                       inst✝¹ : (i : ι) → Group (G i)
                                                       inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
                                                       i : ι
                                                       x✝ : G i
                                                       ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) 0 (Quoti …
                                                     -/
  zpow_zero' := DirectLimit.induction _ fun i _ ↦ by simp_rw [map_def, zpow_zero, one_def i]
                                                     /-
                                                       🎉 no goals
                                                     -/
  zpow_succ' n := DirectLimit.induction _ fun i x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → Group (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
      n : Nat
      i : ι
      x : G i
      ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) (↑n.succ …
    -/
    simp_rw [map_def, mul_def]; congr; apply DivInvMonoid.zpow_succ'
                                       /-
                                         🎉 no goals
                                       -/
  zpow_neg' n := DirectLimit.induction _ fun i x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → Group (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
      n : Nat
      i : ι
      x : G i
      ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) (Int.neg …
    -/
    simp_rw [map_def]; congr; apply DivInvMonoid.zpow_neg'
                              /-
                                🎉 no goals
                              -/
  inv_mul_cancel := DirectLimit.induction _ fun i _ ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → Group (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
      i : ι
      x✝ : G i
      ⊢ Eq (HMul.hMul (Inv.inv (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩)) (Quotie …
    -/
    simp_rw [map_def, mul_def, inv_mul_cancel, one_def i]
    /-
      🎉 no goals
    -/


@[to_additive] theorem inv_def (i x) : (⟦⟨i, x⟩⟧)⁻¹ = (⟦⟨i, x⁻¹⟩⟧ : DirectLimit G f) := rfl


@[to_additive] theorem div_def (i x y) : ⟦⟨i, x⟩⟧ / ⟦⟨i, y⟩⟧ = (⟦⟨i, x / y⟩⟧ : DirectLimit G f) :=
  map₂_def ..


@[to_additive] theorem zpow_def (i x) (n : ℤ) : ⟦⟨i, x⟩⟧ ^ n = (⟦⟨i, x ^ n⟩⟧ : DirectLimit G f) :=
  rfl


@[to_additive] instance [∀ i, CommGroup (G i)] [∀ i j h, MonoidHomClass (T h) (G i) (G j)] :
    CommGroup (DirectLimit G f) where
  mul_comm := mul_comm


instance [∀ i, MulZeroClass (G i)] [∀ i j h, MulHomClass (T h) (G i) (G j)]
    [∀ i j h, ZeroHomClass (T h) (G i) (G j)] :
    MulZeroClass (DirectLimit G f) where
                                                   /-
                                                     R : Type u_1
                                                     ι : Type u_2
                                                     inst✝⁷ : Preorder ι
                                                     G : ι → Type u_3
                                                     T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                     f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                     inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                     inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                     inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                     inst✝³ : Nonempty ι
                                                     inst✝² : (i : ι) → MulZeroClass (G i)
                                                     inst✝¹ : ∀ (i j : ι) (h : LE.le i j), MulHomClass (T h) (G i) (G j)
                                                     inst✝ : ∀ (i j : ι) (h : LE.le i j), ZeroHomClass (T h) (G i) (G j)
                                                     i : ι
                                                     x✝ : G i
                                                     ⊢ Eq (HMul.hMul 0 (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩)) 0
                                                   -/
  zero_mul := DirectLimit.induction _ fun i _ ↦ by simp_rw [zero_def i, mul_def, zero_mul]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     R : Type u_1
                                                     ι : Type u_2
                                                     inst✝⁷ : Preorder ι
                                                     G : ι → Type u_3
                                                     T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                     f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                     inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                     inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                     inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                     inst✝³ : Nonempty ι
                                                     inst✝² : (i : ι) → MulZeroClass (G i)
                                                     inst✝¹ : ∀ (i j : ι) (h : LE.le i j), MulHomClass (T h) (G i) (G j)
                                                     inst✝ : ∀ (i j : ι) (h : LE.le i j), ZeroHomClass (T h) (G i) (G j)
                                                     i : ι
                                                     x✝ : G i
                                                     ⊢ Eq (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩) 0) 0
                                                   -/
  mul_zero := DirectLimit.induction _ fun i _ ↦ by simp_rw [zero_def i, mul_def, mul_zero]
                                                   /-
                                                     🎉 no goals
                                                   -/


instance : MulZeroOneClass (DirectLimit G f) where
  zero_mul := zero_mul
  mul_zero := mul_zero


instance [∀ i, Nontrivial (G i)] : Nontrivial (DirectLimit G f) where
                                                                              /-
                                                                                R : Type u_1
                                                                                ι : Type u_2
                                                                                inst✝⁷ : Preorder ι
                                                                                G : ι → Type u_3
                                                                                T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                                                f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                                                inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                                                inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                                                inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                                                inst✝³ : Nonempty ι
                                                                                inst✝² : (i : ι) → MulZeroOneClass (G i)
                                                                                inst✝¹ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
                                                                                inst✝ : ∀ (i : ι), Nontrivial (G i)
                                                                                h : Eq 0 1
                                                                                i : ι
                                                                                w✝¹ : LE.le ⟨Classical.arbitrary ι, (fun x => 0) (Classical.arbitrary ι)⟩.fst i
                                                                                w✝ : LE.le ⟨Classical.arbitrary ι, (fun x => 1) (Classical.arbitrary ι)⟩.fst i
                                                                                eq : Eq ((f ⟨Classical.arbitrary ι, (fun x => 0) (Classical.arbitrary ι)⟩.fst  …
                                                                                ⊢ False
                                                                              -/
  exists_pair_ne := ⟨0, 1, fun h ↦ have ⟨i, _, _, eq⟩ := Quotient.eq.mp h; by simp at eq⟩
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance [∀ i, SemigroupWithZero (G i)] [∀ i j h, MulHomClass (T h) (G i) (G j)]
    [∀ i j h, ZeroHomClass (T h) (G i) (G j)] :
    SemigroupWithZero (DirectLimit G f) where
  zero_mul := zero_mul
  mul_zero := mul_zero


instance [∀ i, MonoidWithZero (G i)] [∀ i j h, MonoidWithZeroHomClass (T h) (G i) (G j)] :
    MonoidWithZero (DirectLimit G f) where
  zero_mul := zero_mul
  mul_zero := mul_zero


instance [∀ i, CommMonoidWithZero (G i)] [∀ i j h, MonoidWithZeroHomClass (T h) (G i) (G j)] :
    CommMonoidWithZero (DirectLimit G f) where
  zero_mul := zero_mul
  mul_zero := mul_zero


instance : GroupWithZero (DirectLimit G f) where
  inv := map _ _ (fun _ ↦ (·⁻¹)) fun _ _ _ ↦ map_inv₀ _
  div := map₂ _ _ _ (fun _ ↦ (· / ·)) fun _ _ _ ↦ map_div₀ _
  zpow n := map _ _ (fun _ ↦ (· ^ n)) fun _ _ _ x ↦ map_zpow₀ _ x n
  div_eq_mul_inv := DirectLimit.induction₂ _ fun i _ _ ↦ show map₂ .. = _ * map .. by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → GroupWithZero (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
      i : ι
      x✝¹ x✝ : G i
      ⊢ Eq (DirectLimit.map₂ f f f (fun x x1 x2 => HDiv.hDiv x1 x2) ⋯ (Quotient.mk ( …
    -/
    simp_rw [map₂_def, map_def, div_eq_mul_inv, mul_def]
    /-
      🎉 no goals
    -/
                                                     /-
                                                       R : Type u_1
                                                       ι : Type u_2
                                                       inst✝⁶ : Preorder ι
                                                       G : ι → Type u_3
                                                       T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                       f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                       inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                       inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                       inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                       inst✝² : Nonempty ι
                                                       inst✝¹ : (i : ι) → GroupWithZero (G i)
                                                       inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
                                                       i : ι
                                                       x✝ : G i
                                                       ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) 0 (Quoti …
                                                     -/
  zpow_zero' := DirectLimit.induction _ fun i _ ↦ by simp_rw [map_def, zpow_zero, one_def i]
                                                     /-
                                                       🎉 no goals
                                                     -/
  zpow_succ' n := DirectLimit.induction _ fun i x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → GroupWithZero (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
      n : Nat
      i : ι
      x : G i
      ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) (↑n.succ …
    -/
    simp_rw [map_def, mul_def]; congr; apply DivInvMonoid.zpow_succ'
                                       /-
                                         🎉 no goals
                                       -/
  zpow_neg' n := DirectLimit.induction _ fun i x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → GroupWithZero (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
      n : Nat
      i : ι
      x : G i
      ⊢ Eq ((fun n => DirectLimit.map f f (fun x x_1 => HPow.hPow x_1 n) ⋯) (Int.neg …
    -/
    simp_rw [map_def]; congr; apply DivInvMonoid.zpow_neg'
                              /-
                                🎉 no goals
                              -/
                                /-
                                  R : Type u_1
                                  ι : Type u_2
                                  inst✝⁶ : Preorder ι
                                  G : ι → Type u_3
                                  T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                  f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                  inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                  inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                  inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                  inst✝² : Nonempty ι
                                  inst✝¹ : (i : ι) → GroupWithZero (G i)
                                  inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
                                  ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨⟨Classical.arbitrary ι, (fun x => 0) …
                                -/
  inv_zero := show ⟦_⟧ = ⟦_⟧ by simp_rw [inv_zero]
                                /-
                                  🎉 no goals
                                -/
  mul_inv_cancel := DirectLimit.induction _ fun i x ne ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → GroupWithZero (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
      i : ι
      x : G i
      ne : Ne (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) 0
      ⊢ Eq (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Inv.inv (Quotient …
    -/
    have : x ≠ 0 := by rintro rfl; exact ne (zero_def i).symm
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → GroupWithZero (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MonoidWithZeroHomClass (T h) (G i) (G j)
      i : ι
      x : G i
      ne : Ne (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) 0
      this : Ne x 0
      ⊢ Eq (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Inv.inv (Quotient …
    -/
    simp_rw [map_def, mul_def, mul_inv_cancel₀ this, one_def i]
    /-
      🎉 no goals
    -/


theorem inv₀_def (i x) : (⟦⟨i, x⟩⟧)⁻¹ = (⟦⟨i, x⁻¹⟩⟧ : DirectLimit G f) := rfl


theorem div₀_def (i x y) : ⟦⟨i, x⟩⟧ / ⟦⟨i, y⟩⟧ = (⟦⟨i, x / y⟩⟧ : DirectLimit G f) :=
  map₂_def ..


theorem zpow₀_def (i x) (n : ℤ) : ⟦⟨i, x⟩⟧ ^ n = (⟦⟨i, x ^ n⟩⟧ : DirectLimit G f) := rfl


instance [∀ i, CommGroupWithZero (G i)] [∀ i j h, MonoidWithZeroHomClass (T h) (G i) (G j)] :
    CommGroupWithZero (DirectLimit G f) where
  __ : GroupWithZero _ := inferInstance
  mul_comm := mul_comm


instance : AddMonoidWithOne (DirectLimit G f) where
  natCast n := map₀ _ fun _ ↦ n
                                    /-
                                      R : Type u_1
                                      ι : Type u_2
                                      inst✝⁶ : Preorder ι
                                      G : ι → Type u_3
                                      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                      inst✝² : Nonempty ι
                                      inst✝¹ : (i : ι) → AddMonoidWithOne (G i)
                                      inst✝ : ∀ (i j : ι) (h : LE.le i j), AddMonoidHomClass (T h) (G i) (G j)
                                      ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, (fun x => ↑0) …
                                    -/
  natCast_zero := show ⟦_⟧ = ⟦_⟧ by simp_rw [Nat.cast_zero]
                                    /-
                                      🎉 no goals
                                    -/
                                            /-
                                              R : Type u_1
                                              ι : Type u_2
                                              inst✝⁶ : Preorder ι
                                              G : ι → Type u_3
                                              T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                              f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                              inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                              inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                              inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                              inst✝² : Nonempty ι
                                              inst✝¹ : (i : ι) → AddMonoidWithOne (G i)
                                              inst✝ : ∀ (i j : ι) (h : LE.le i j), AddMonoidHomClass (T h) (G i) (G j)
                                              n : Nat
                                              ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, (fun x => ↑(H …
                                            -/
  natCast_succ n := show ⟦_⟧ = ⟦_⟧ + ⟦_⟧ by simp_rw [Nat.cast_succ, add_def]
                                            /-
                                              🎉 no goals
                                            -/


theorem natCast_def [∀ i j h, OneHomClass (T h) (G i) (G j)] (n : ℕ) (i) :
    (n : DirectLimit G f) = ⟦⟨i, n⟩⟧ :=
  map₀_def _ _ (fun _ _ _ ↦ map_natCast' _ (map_one _) _) _


instance : AddGroupWithOne (DirectLimit G f) where
  __ : AddGroup _ := inferInstance
  intCast n := map₀ _ fun _ ↦ n
                                       /-
                                         R : Type u_1
                                         ι : Type u_2
                                         inst✝⁶ : Preorder ι
                                         G : ι → Type u_3
                                         T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                         f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                         inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                         inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                         inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                         inst✝² : Nonempty ι
                                         inst✝¹ : (i : ι) → AddGroupWithOne (G i)
                                         inst✝ : ∀ (i j : ι) (h : LE.le i j), AddMonoidHomClass (T h) (G i) (G j)
                                         n : Nat
                                         ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, (fun x => ↑↑n …
                                       -/
  intCast_ofNat n := show ⟦_⟧ = ⟦_⟧ by simp_rw [Int.cast_natCast]
                                       /-
                                         🎉 no goals
                                       -/
                                         /-
                                           R : Type u_1
                                           ι : Type u_2
                                           inst✝⁶ : Preorder ι
                                           G : ι → Type u_3
                                           T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                           f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                           inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                           inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                           inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                           inst✝² : Nonempty ι
                                           inst✝¹ : (i : ι) → AddGroupWithOne (G i)
                                           inst✝ : ∀ (i j : ι) (h : LE.le i j), AddMonoidHomClass (T h) (G i) (G j)
                                           n : Nat
                                           ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, (fun x => ↑(I …
                                         -/
  intCast_negSucc n := show ⟦_⟧ = ⟦_⟧ by simp
                                         /-
                                           🎉 no goals
                                         -/
  natCast_zero := Nat.cast_zero
  natCast_succ := Nat.cast_succ


theorem intCast_def [∀ i j h, OneHomClass (T h) (G i) (G j)] (n : ℤ) (i) :
    (n : DirectLimit G f) = ⟦⟨i, n⟩⟧ :=
  map₀_def _ _ (fun _ _ _ ↦ map_intCast' _ (map_one _) _) _


instance [∀ i, AddCommMonoidWithOne (G i)] [∀ i j h, AddMonoidHomClass (T h) (G i) (G j)] :
    AddCommMonoidWithOne (DirectLimit G f) where
  add_comm := add_comm


instance [∀ i, AddCommGroupWithOne (G i)] [∀ i j h, AddMonoidHomClass (T h) (G i) (G j)] :
    AddCommGroupWithOne (DirectLimit G f) where
  __ : AddGroupWithOne _ := inferInstance
  add_comm := add_comm


instance [∀ i, NonUnitalNonAssocSemiring (G i)] [∀ i j h, NonUnitalRingHomClass (T h) (G i) (G j)] :
    NonUnitalNonAssocSemiring (DirectLimit G f) where
  left_distrib := DirectLimit.induction₃ _ fun i _ _ _ ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → NonUnitalNonAssocSemiring (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), NonUnitalRingHomClass (T h) (G i) (G j)
      i : ι
      x✝² x✝¹ x✝ : G i
      ⊢ Eq (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝²⟩) (HAdd.hAdd (Quot …
    -/
    simp_rw [add_def, mul_def, left_distrib, add_def]
    /-
      🎉 no goals
    -/
  right_distrib := DirectLimit.induction₃ _ fun i _ _ _ ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → NonUnitalNonAssocSemiring (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), NonUnitalRingHomClass (T h) (G i) (G j)
      i : ι
      x✝² x✝¹ x✝ : G i
      ⊢ Eq (HMul.hMul (HAdd.hAdd (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝²⟩) (Quot …
    -/
    simp_rw [add_def, mul_def, right_distrib, add_def]
    /-
      🎉 no goals
    -/
  zero_mul := zero_mul
  mul_zero := mul_zero


instance [∀ i, NonUnitalNonAssocCommSemiring (G i)]
    [∀ i j h, NonUnitalRingHomClass (T h) (G i) (G j)] :
    NonUnitalNonAssocCommSemiring (DirectLimit G f) where
  mul_comm := mul_comm


instance [∀ i, NonUnitalSemiring (G i)] [∀ i j h, NonUnitalRingHomClass (T h) (G i) (G j)] :
    NonUnitalSemiring (DirectLimit G f) where
  mul_assoc := mul_assoc


instance [∀ i, NonUnitalCommSemiring (G i)] [∀ i j h, NonUnitalRingHomClass (T h) (G i) (G j)] :
    NonUnitalCommSemiring (DirectLimit G f) where
  mul_comm := mul_comm


instance [∀ i, NonAssocSemiring (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] :
    NonAssocSemiring (DirectLimit G f) where
  one_mul := one_mul
  mul_one := mul_one
  natCast_zero := Nat.cast_zero
  natCast_succ := Nat.cast_succ

-- There is no NonAssocCommSemiring


instance [∀ i, Semiring (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] :
    Semiring (DirectLimit G f) where
  __ : NonAssocSemiring _ := inferInstance
  __ : Monoid _ := inferInstance


instance [∀ i, CommSemiring (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] :
    CommSemiring (DirectLimit G f) where
  mul_comm := mul_comm


instance [∀ i, Ring (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] : Ring (DirectLimit G f) where
  __ : Semiring _ := inferInstance
  __ : AddCommGroupWithOne _ := inferInstance


instance [∀ i, CommRing (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] :
    CommRing (DirectLimit G f) where
  mul_comm := mul_comm


instance [∀ i, Zero (G i)] [∀ i, SMulZeroClass R (G i)]
    [∀ i j h, MulActionHomClass (T h) R (G i) (G j)] :
    SMulZeroClass R (DirectLimit G f) where
                                              /-
                                                R : Type u_1
                                                ι : Type u_2
                                                inst✝⁷ : Preorder ι
                                                G : ι → Type u_3
                                                T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                inst✝³ : Nonempty ι
                                                inst✝² : (i : ι) → Zero (G i)
                                                inst✝¹ : (i : ι) → SMulZeroClass R (G i)
                                                inst✝ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
                                                r : R
                                                ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, HSMul.hSMul r …
                                              -/
  smul_zero r := (smul_def _ _ _).trans <| by rw [smul_zero]; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


instance [Zero R] [∀ i, Zero (G i)] [∀ i, SMulWithZero R (G i)]
    [∀ i j h, MulActionHomClass (T h) R (G i) (G j)]
    [∀ i j h, ZeroHomClass (T h) (G i) (G j)] :
    SMulWithZero R (DirectLimit G f) where
                                                    /-
                                                      R : Type u_1
                                                      ι : Type u_2
                                                      inst✝⁹ : Preorder ι
                                                      G : ι → Type u_3
                                                      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                      inst✝⁸ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                      inst✝⁷ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                      inst✝⁶ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                      inst✝⁵ : Nonempty ι
                                                      inst✝⁴ : Zero R
                                                      inst✝³ : (i : ι) → Zero (G i)
                                                      inst✝² : (i : ι) → SMulWithZero R (G i)
                                                      inst✝¹ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
                                                      inst✝ : ∀ (i j : ι) (h : LE.le i j), ZeroHomClass (T h) (G i) (G j)
                                                      i : ι
                                                      x✝ : G i
                                                      ⊢ Eq (HSMul.hSMul 0 (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩)) 0
                                                    -/
  zero_smul := DirectLimit.induction _ fun i _ ↦ by simp_rw [smul_def, zero_smul, zero_def i]
                                                    /-
                                                      🎉 no goals
                                                    -/


instance [∀ i, AddZeroClass (G i)] [∀ i, DistribSMul R (G i)]
    [∀ i j h, AddMonoidHomClass (T h) (G i) (G j)]
    [∀ i j h, MulActionHomClass (T h) R (G i) (G j)] :
    DistribSMul R (DirectLimit G f) where
  smul_add r := DirectLimit.induction₂ _ fun i _ _ ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁸ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁷ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁶ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝⁵ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝⁴ : Nonempty ι
      inst✝³ : (i : ι) → AddZeroClass (G i)
      inst✝² : (i : ι) → DistribSMul R (G i)
      inst✝¹ : ∀ (i j : ι) (h : LE.le i j), AddMonoidHomClass (T h) (G i) (G j)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
      r : R
      i : ι
      x✝¹ x✝ : G i
      ⊢ Eq (HSMul.hSMul r (HAdd.hAdd (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝¹⟩) ( …
    -/
    simp_rw [add_def, smul_def, smul_add, add_def]
    /-
      🎉 no goals
    -/


instance [Monoid R] [∀ i, AddMonoid (G i)] [∀ i, DistribMulAction R (G i)]
    [∀ i j h, DistribMulActionHomClass (T h) R (G i) (G j)] :
    DistribMulAction R (DirectLimit G f) :=
  have _ i j h : MulActionHomClass (T h) R (G i) (G j) := inferInstance
  { smul_zero := smul_zero, smul_add := smul_add }


instance [Monoid R] [∀ i, Monoid (G i)] [∀ i, MulDistribMulAction R (G i)]
    [∀ i j h, MonoidHomClass (T h) (G i) (G j)]
    [∀ i j h, MulActionHomClass (T h) R (G i) (G j)] :
    MulDistribMulAction R (DirectLimit G f) where
  smul_mul r := DirectLimit.induction₂ _ fun i _ _ ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁹ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁸ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁷ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝⁶ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝⁵ : Nonempty ι
      inst✝⁴ : Monoid R
      inst✝³ : (i : ι) → Monoid (G i)
      inst✝² : (i : ι) → MulDistribMulAction R (G i)
      inst✝¹ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
      r : R
      i : ι
      x✝¹ x✝ : G i
      ⊢ Eq (HSMul.hSMul r (HMul.hMul (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝¹⟩) ( …
    -/
    simp_rw [mul_def, smul_def, MulDistribMulAction.smul_mul, mul_def]
    /-
      🎉 no goals
    -/
                                             /-
                                               R : Type u_1
                                               ι : Type u_2
                                               inst✝⁹ : Preorder ι
                                               G : ι → Type u_3
                                               T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                               f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                               inst✝⁸ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                               inst✝⁷ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                               inst✝⁶ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                               inst✝⁵ : Nonempty ι
                                               inst✝⁴ : Monoid R
                                               inst✝³ : (i : ι) → Monoid (G i)
                                               inst✝² : (i : ι) → MulDistribMulAction R (G i)
                                               inst✝¹ : ∀ (i j : ι) (h : LE.le i j), MonoidHomClass (T h) (G i) (G j)
                                               inst✝ : ∀ (i j : ι) (h : LE.le i j), MulActionHomClass (T h) R (G i) (G j)
                                               r : R
                                               ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, HSMul.hSMul r …
                                             -/
  smul_one r := (smul_def _ _ _).trans <| by rw [smul_one]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


instance [Semiring R] [∀ i, AddCommMonoid (G i)] [∀ i, Module R (G i)]
    [∀ i j h, LinearMapClass (T h) R (G i) (G j)] :
    Module R (DirectLimit G f) :=
  have _ i j h : DistribMulActionHomClass (T h) R (G i) (G j) := inferInstance
                                                         /-
                                                           R : Type u_1
                                                           ι : Type u_2
                                                           inst✝⁸ : Preorder ι
                                                           G : ι → Type u_3
                                                           T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                           f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                           inst✝⁷ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                           inst✝⁶ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                           inst✝⁵ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                           inst✝⁴ : Nonempty ι
                                                           inst✝³ : Semiring R
                                                           inst✝² : (i : ι) → AddCommMonoid (G i)
                                                           inst✝¹ : (i : ι) → Module R (G i)
                                                           inst✝ : ∀ (i j : ι) (h : LE.le i j), LinearMapClass (T h) R (G i) (G j)
                                                           x✝³ : ∀ (i j : ι) (h : LE.le i j), DistribMulActionHomClass (T h) R (G i) (G j)
                                                           x✝² x✝¹ : R
                                                           i : ι
                                                           x✝ : G i
                                                           ⊢ Eq (HSMul.hSMul (HAdd.hAdd x✝² x✝¹) (Quotient.mk (DirectLimit.setoid f) ⟨i,  …
                                                         -/
  { add_smul _ _ := DirectLimit.induction _ fun i _ ↦ by simp_rw [smul_def, add_smul, add_def],
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                      /-
                                                        R : Type u_1
                                                        ι : Type u_2
                                                        inst✝⁸ : Preorder ι
                                                        G : ι → Type u_3
                                                        T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                        f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                        inst✝⁷ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                        inst✝⁶ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                        inst✝⁵ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                        inst✝⁴ : Nonempty ι
                                                        inst✝³ : Semiring R
                                                        inst✝² : (i : ι) → AddCommMonoid (G i)
                                                        inst✝¹ : (i : ι) → Module R (G i)
                                                        inst✝ : ∀ (i j : ι) (h : LE.le i j), LinearMapClass (T h) R (G i) (G j)
                                                        x✝¹ : ∀ (i j : ι) (h : LE.le i j), DistribMulActionHomClass (T h) R (G i) (G j)
                                                        i : ι
                                                        x✝ : G i
                                                        ⊢ Eq (HSMul.hSMul 0 (Quotient.mk (DirectLimit.setoid f) ⟨i, x✝⟩)) 0
                                                      -/
    zero_smul := DirectLimit.induction _ fun i _ ↦ by simp_rw [smul_def, zero_smul, zero_def i] }
                                                      /-
                                                        🎉 no goals
                                                      -/


instance : DivisionSemiring (DirectLimit G f) where
  __ : GroupWithZero _ := inferInstance
  __ : Semiring _ := inferInstance
  nnratCast q := map₀ _ fun _ ↦ q
                                             /-
                                               R : Type u_1
                                               ι : Type u_2
                                               inst✝⁶ : Preorder ι
                                               G : ι → Type u_3
                                               T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                               f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                               inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                               inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                               inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                               inst✝² : Nonempty ι
                                               inst✝¹ : (i : ι) → DivisionSemiring (G i)
                                               inst✝ : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
                                               q : NNRat
                                               ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, (fun x => ↑q) …
                                             -/
  nnratCast_def q := show ⟦_⟧ = ⟦_⟧ / ⟦_⟧ by simp_rw [div₀_def]; rw [NNRat.cast_def]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  nnqsmul q := map _ _ (fun _ ↦ (q • ·)) fun _ _ _ x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → DivisionSemiring (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
      q : NNRat
      x✝² x✝¹ : ι
      x✝ : LE.le x✝² x✝¹
      x : G x✝²
      ⊢ Eq ((f x✝² x✝¹ x✝) ((fun x x_1 => HSMul.hSMul q x_1) x✝² x)) ((fun x x_1 =>  …
    -/
    simp_rw [NNRat.smul_def, map_mul, map_nnratCast]
    /-
      🎉 no goals
    -/
  nnqsmul_def _ := DirectLimit.induction _ fun i x ↦ show ⟦_⟧ = map₀ .. * _ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → DivisionSemiring (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
      x✝ : NNRat
      i : ι
      x : G i
      ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨⟨i, x⟩.fst, (fun x x_1 => HSMul.hSMu …
    -/
    simp_rw [map₀_def _ _ (fun _ _ _ ↦ map_nnratCast _ _) i, mul_def, NNRat.smul_def]
    /-
      🎉 no goals
    -/


theorem nnratCast_def (q : ℚ≥0) (i) : (q : DirectLimit G f) = ⟦⟨i, q⟩⟧ :=
  map₀_def _ _ (fun _ _ _ ↦ map_nnratCast _ _) _


instance [∀ i, Semifield (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] :
    Semifield (DirectLimit G f) where
  __ : DivisionSemiring _ := inferInstance
  mul_comm := mul_comm


instance : DivisionRing (DirectLimit G f) where
  __ : DivisionSemiring _ := inferInstance
  __ : Ring _ := inferInstance
  ratCast q := map₀ _ fun _ ↦ q
                                           /-
                                             R : Type u_1
                                             ι : Type u_2
                                             inst✝⁶ : Preorder ι
                                             G : ι → Type u_3
                                             T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                             f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                             inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                             inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                             inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                             inst✝² : Nonempty ι
                                             inst✝¹ : (i : ι) → DivisionRing (G i)
                                             inst✝ : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
                                             q : Rat
                                             ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨Classical.arbitrary ι, (fun x => ↑q) …
                                           -/
  ratCast_def q := show ⟦_⟧ = ⟦_⟧ / ⟦_⟧ by simp_rw [div₀_def]; rw [Rat.cast_def]
                                                               /-
                                                                 🎉 no goals
                                                               -/
  qsmul q := map _ _ (fun _ ↦ (q • ·)) fun _ _ _ x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → DivisionRing (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
      q : Rat
      x✝² x✝¹ : ι
      x✝ : LE.le x✝² x✝¹
      x : G x✝²
      ⊢ Eq ((f x✝² x✝¹ x✝) ((fun x x_1 => HSMul.hSMul q x_1) x✝² x)) ((fun x x_1 =>  …
    -/
    simp_rw [Rat.smul_def, map_mul, map_ratCast]
    /-
      🎉 no goals
    -/
  qsmul_def _ := DirectLimit.induction _ fun i x ↦ show ⟦_⟧ = map₀ .. * _ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁵ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁴ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      inst✝¹ : (i : ι) → DivisionRing (G i)
      inst✝ : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
      x✝ : Rat
      i : ι
      x : G i
      ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨⟨i, x⟩.fst, (fun x x_1 => HSMul.hSMu …
    -/
    simp_rw [map₀_def _ _ (fun _ _ _ ↦ map_ratCast _ _) i, mul_def, Rat.smul_def]
    /-
      🎉 no goals
    -/


theorem ratCast_def (q : ℚ) (i) : (q : DirectLimit G f) = ⟦⟨i, q⟩⟧ :=
  map₀_def _ _ (fun _ _ _ ↦ map_ratCast _ _) _


instance [∀ i, Field (G i)] [∀ i j h, RingHomClass (T h) (G i) (G j)] :
    Field (DirectLimit G f) where
  __ : DivisionRing _ := inferInstance
  mul_comm := mul_comm


/-- The canonical map from a component to the direct limit. -/
def of (i) : G i →ₗ[R] DirectLimit G f where
  toFun x := ⟦⟨i, x⟩⟧
  map_add' _ _ := (add_def ..).symm
  map_smul' _ _ := (smul_def ..).symm


@[simp]
theorem of_f {i j hij x} : of R ι G f j (f i j hij x) = of R ι G f i x := .symm <| eq_of_le ..


variable (R ι G f) in
/-- The universal property of the direct limit: maps from the components to another module
that respect the directed system structure (i.e. make some diagram commute) give rise
to a unique map out of the direct limit. -/
def lift (g : ∀ i, G i →ₗ[R] P) (Hg : ∀ i j hij x, g j (f i j hij x) = g i x) :
    DirectLimit G f →ₗ[R] P where
  toFun := _root_.DirectLimit.lift _ (g · ·) fun i j h x ↦ (Hg i j h x).symm
                                                      /-
                                                        R : Type u_1
                                                        ι : Type u_2
                                                        inst✝¹⁰ : Preorder ι
                                                        G : ι → Type u_3
                                                        T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                        f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                        inst✝⁹ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                        inst✝⁸ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                        inst✝⁷ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                        inst✝⁶ : Semiring R
                                                        inst✝⁵ : (i : ι) → AddCommMonoid (G i)
                                                        inst✝⁴ : (i : ι) → Module R (G i)
                                                        inst✝³ : ∀ (i j : ι) (h : LE.le i j), LinearMapClass (T h) R (G i) (G j)
                                                        inst✝² : Nonempty ι
                                                        P : Type u_5
                                                        inst✝¹ : AddCommMonoid P
                                                        inst✝ : Module R P
                                                        g : (i : ι) → LinearMap (RingHom.id R) (G i) P
                                                        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                                        i : ι
                                                        x y : G i
                                                        ⊢ Eq (DirectLimit.lift f (fun x1 x2 => (g x1) x2) ⋯ (HAdd.hAdd (Quotient.mk (D …
                                                      -/
  map_add' := DirectLimit.induction₂ _ fun i x y ↦ by simp_rw [add_def, lift_def, map_add]
                                                      /-
                                                        🎉 no goals
                                                      -/
  map_smul' r := DirectLimit.induction _ fun i x ↦ by
    /-
      R : Type u_1
      ι : Type u_2
      inst✝¹⁰ : Preorder ι
      G : ι → Type u_3
      T : ⦃i j : ι⦄ → LE.le i j → Type u_4
      f : (x x_1 : ι) → (h : LE.le x x_1) → T h
      inst✝⁹ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
      inst✝⁸ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝⁷ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → AddCommMonoid (G i)
      inst✝⁴ : (i : ι) → Module R (G i)
      inst✝³ : ∀ (i j : ι) (h : LE.le i j), LinearMapClass (T h) R (G i) (G j)
      inst✝² : Nonempty ι
      P : Type u_5
      inst✝¹ : AddCommMonoid P
      inst✝ : Module R P
      g : (i : ι) → LinearMap (RingHom.id R) (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      r : R
      i : ι
      x : G i
      ⊢ Eq ({ toFun := DirectLimit.lift f (fun x1 x2 => (g x1) x2) ⋯, map_add' := ⋯  …
    -/
    simp_rw [smul_def, lift_def, map_smul, RingHom.id_apply]
    /-
      🎉 no goals
    -/


theorem lift_of {i} (x) : lift R ι G f g Hg (of R ι G f i x) = g i x := rfl


variable (G f) in
/-- The canonical map from a component to the direct limit. -/
nonrec def of (i) : G i →+* DirectLimit G f where
  toFun x := ⟦⟨i, x⟩⟧
  map_one' := (one_def i).symm
  map_mul' _ _ := (mul_def ..).symm
  map_zero' := (zero_def i).symm
  map_add' _ _ := (add_def ..).symm


@[simp] theorem of_f {i j} (hij) (x) : of G f j (f i j hij x) = of G f i x := .symm <| eq_of_le ..


variable (G f) in
/-- The universal property of the direct limit: maps from the components to another ring
that respect the directed system structure (i.e. make some diagram commute) give rise
to a unique map out of the direct limit.
-/
def lift (g : ∀ i, G i →+* P) (Hg : ∀ i j hij x, g j (f i j hij x) = g i x) :
    DirectLimit G f →+* P where
  toFun := _root_.DirectLimit.lift _ (g · ·) fun i j h x ↦ (Hg i j h x).symm
                 /-
                   R : Type u_1
                   ι : Type u_2
                   inst✝⁷ : Preorder ι
                   G : ι → Type u_3
                   T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                   f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                   inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                   inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                   inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                   inst✝³ : (i : ι) → NonAssocSemiring (G i)
                   inst✝² : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
                   inst✝¹ : Nonempty ι
                   P : Type u_5
                   inst✝ : NonAssocSemiring P
                   g : (i : ι) → RingHom (G i) P
                   Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                   ⊢ Eq (DirectLimit.lift f (fun x1 x2 => (g x1) x2) ⋯ 1) 1
                 -/
  map_one' := by rw [one_def (Classical.arbitrary ι), lift_def, map_one]
                 /-
                   🎉 no goals
                 -/
                                                      /-
                                                        R : Type u_1
                                                        ι : Type u_2
                                                        inst✝⁷ : Preorder ι
                                                        G : ι → Type u_3
                                                        T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                        f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                        inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                        inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                        inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                        inst✝³ : (i : ι) → NonAssocSemiring (G i)
                                                        inst✝² : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
                                                        inst✝¹ : Nonempty ι
                                                        P : Type u_5
                                                        inst✝ : NonAssocSemiring P
                                                        g : (i : ι) → RingHom (G i) P
                                                        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                                        i : ι
                                                        x y : G i
                                                        ⊢ Eq ({ toFun := DirectLimit.lift f (fun x1 x2 => (g x1) x2) ⋯, map_one' := ⋯  …
                                                      -/
  map_mul' := DirectLimit.induction₂ _ fun i x y ↦ by simp_rw [mul_def, lift_def, map_mul]
                                                      /-
                                                        🎉 no goals
                                                      -/
                  /-
                    R : Type u_1
                    ι : Type u_2
                    inst✝⁷ : Preorder ι
                    G : ι → Type u_3
                    T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                    f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                    inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                    inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                    inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                    inst✝³ : (i : ι) → NonAssocSemiring (G i)
                    inst✝² : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
                    inst✝¹ : Nonempty ι
                    P : Type u_5
                    inst✝ : NonAssocSemiring P
                    g : (i : ι) → RingHom (G i) P
                    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                    ⊢ Eq ((↑{ toFun := DirectLimit.lift f (fun x1 x2 => (g x1) x2) ⋯, map_one' :=  …
                  -/
  map_zero' := by simp_rw [zero_def (Classical.arbitrary ι), lift_def, map_zero]
                  /-
                    🎉 no goals
                  -/
                                                      /-
                                                        R : Type u_1
                                                        ι : Type u_2
                                                        inst✝⁷ : Preorder ι
                                                        G : ι → Type u_3
                                                        T : ⦃i j : ι⦄ → LE.le i j → Type u_4
                                                        f : (x x_1 : ι) → (h : LE.le x x_1) → T h
                                                        inst✝⁶ : (i j : ι) → (h : LE.le i j) → FunLike (T h) (G i) (G j)
                                                        inst✝⁵ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                        inst✝⁴ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                        inst✝³ : (i : ι) → NonAssocSemiring (G i)
                                                        inst✝² : ∀ (i j : ι) (h : LE.le i j), RingHomClass (T h) (G i) (G j)
                                                        inst✝¹ : Nonempty ι
                                                        P : Type u_5
                                                        inst✝ : NonAssocSemiring P
                                                        g : (i : ι) → RingHom (G i) P
                                                        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                                        i : ι
                                                        x y : G i
                                                        ⊢ Eq ((↑{ toFun := DirectLimit.lift f (fun x1 x2 => (g x1) x2) ⋯, map_one' :=  …
                                                      -/
  map_add' := DirectLimit.induction₂ _ fun i x y ↦ by simp_rw [add_def, lift_def, map_add]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] theorem lift_of (i x) : lift G f P g Hg (of G f i x) = g i x := rfl


