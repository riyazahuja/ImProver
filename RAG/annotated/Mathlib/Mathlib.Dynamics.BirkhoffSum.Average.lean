/-- The average value of `g` on the first `n` points of the orbit of `x` under `f`,
i.e. the Birkhoff sum `∑ k ∈ Finset.range n, g (f^[k] x)` divided by `n`.

This average appears in many ergodic theorems
which say that `(birkhoffAverage R f g · x)`
converges to the "space average" `⨍ x, g x ∂μ` as `n → ∞`.

We use an auxiliary `[DivisionSemiring R]` to define division by `n`.
However, the definition does not depend on the choice of `R`,
see `birkhoffAverage_congr_ring`. -/
def birkhoffAverage (f : α → α) (g : α → M) (n : ℕ) (x : α) : M := (n : R)⁻¹ • birkhoffSum f g n x


theorem birkhoffAverage_zero (f : α → α) (g : α → M) (x : α) :
                                        /-
                                          R : Type u_1
                                          α : Type u_2
                                          M : Type u_3
                                          inst✝² : DivisionSemiring R
                                          inst✝¹ : AddCommMonoid M
                                          inst✝ : Module R M
                                          f : α → α
                                          g : α → M
                                          x : α
                                          ⊢ Eq (birkhoffAverage R f g 0 x) 0
                                        -/
    birkhoffAverage R f g 0 x = 0 := by simp [birkhoffAverage]
                                        /-
                                          🎉 no goals
                                        -/


@[simp] theorem birkhoffAverage_zero' (f : α → α) (g : α → M) : birkhoffAverage R f g 0 = 0 :=
  funext <| birkhoffAverage_zero _ _ _


theorem birkhoffAverage_one (f : α → α) (g : α → M) (x : α) :
                                          /-
                                            R : Type u_1
                                            α : Type u_2
                                            M : Type u_3
                                            inst✝² : DivisionSemiring R
                                            inst✝¹ : AddCommMonoid M
                                            inst✝ : Module R M
                                            f : α → α
                                            g : α → M
                                            x : α
                                            ⊢ Eq (birkhoffAverage R f g 1 x) (g x)
                                          -/
    birkhoffAverage R f g 1 x = g x := by simp [birkhoffAverage]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem birkhoffAverage_one' (f : α → α) (g : α → M) : birkhoffAverage R f g 1 = g :=
  funext <| birkhoffAverage_one R f g


theorem map_birkhoffAverage (S : Type*) {F N : Type*}
    [DivisionSemiring S] [AddCommMonoid N] [Module S N] [FunLike F M N]
    [AddMonoidHomClass F M N] (g' : F) (f : α → α) (g : α → M) (n : ℕ) (x : α) :
    g' (birkhoffAverage R f g n x) = birkhoffAverage S f (g' ∘ g) n x := by
  /-
    R : Type u_1
    α : Type u_2
    M : Type u_3
    inst✝⁷ : DivisionSemiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    S : Type u_4
    F : Type u_5
    N : Type u_6
    inst✝⁴ : DivisionSemiring S
    inst✝³ : AddCommMonoid N
    inst✝² : Module S N
    inst✝¹ : FunLike F M N
    inst✝ : AddMonoidHomClass F M N
    g' : F
    f : α → α
    g : α → M
    n : Nat
    x : α
    ⊢ Eq (g' (birkhoffAverage R f g n x)) (birkhoffAverage S f (Function.comp (⇑g' …
  -/
  simp only [birkhoffAverage, map_inv_natCast_smul g' R S, map_birkhoffSum]
  /-
    🎉 no goals
  -/


theorem birkhoffAverage_congr_ring (S : Type*) [DivisionSemiring S] [Module S M]
    (f : α → α) (g : α → M) (n : ℕ) (x : α) :
    birkhoffAverage R f g n x = birkhoffAverage S f g n x :=
  map_birkhoffAverage R S (AddMonoidHom.id M) f g n x


theorem birkhoffAverage_congr_ring' (S : Type*) [DivisionSemiring S] [Module S M] :
    birkhoffAverage (α := α) (M := M) R = birkhoffAverage S := by
  /-
    R : Type u_1
    α : Type u_2
    M : Type u_3
    inst✝⁴ : DivisionSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_4
    inst✝¹ : DivisionSemiring S
    inst✝ : Module S M
    ⊢ Eq (birkhoffAverage R) (birkhoffAverage S)
  -/
  ext; apply birkhoffAverage_congr_ring
       /-
         🎉 no goals
       -/


theorem Function.IsFixedPt.birkhoffAverage_eq [CharZero R] {f : α → α} {x : α} (h : IsFixedPt f x)
    (g : α → M) {n : ℕ} (hn : n ≠ 0) : birkhoffAverage R f g n x = g x := by
  /-
    R : Type u_1
    α : Type u_2
    M : Type u_3
    inst✝³ : DivisionSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : CharZero R
    f : α → α
    x : α
    h : Function.IsFixedPt f x
    g : α → M
    n : Nat
    hn : Ne n 0
    ⊢ Eq (birkhoffAverage R f g n x) (g x)
  -/
  rw [birkhoffAverage, h.birkhoffSum_eq, ← Nat.cast_smul_eq_nsmul R, inv_smul_smul₀]
  /-
    case ha
    R : Type u_1
    α : Type u_2
    M : Type u_3
    inst✝³ : DivisionSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : CharZero R
    f : α → α
    x : α
    h : Function.IsFixedPt f x
    g : α → M
    n : Nat
    hn : Ne n 0
    ⊢ Ne (↑n) 0
  -/
  rwa [Nat.cast_ne_zero]
  /-
    🎉 no goals
  -/


/-- Birkhoff average is "almost invariant" under `f`:
the difference between `birkhoffAverage R f g n (f x)` and `birkhoffAverage R f g n x`
is equal to `(n : R)⁻¹ • (g (f^[n] x) - g x)`. -/
theorem birkhoffAverage_apply_sub_birkhoffAverage {α M : Type*} (R : Type*) [DivisionRing R]
    [AddCommGroup M] [Module R M] (f : α → α) (g : α → M) (n : ℕ) (x : α) :
    birkhoffAverage R f g n (f x) - birkhoffAverage R f g n x =
      (n : R)⁻¹ • (g (f^[n] x) - g x) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_3
    inst✝² : DivisionRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : α → α
    g : α → M
    n : Nat
    x : α
    ⊢ Eq (HSub.hSub (birkhoffAverage R f g n (f x)) (birkhoffAverage R f g n x)) ( …
  -/
  simp only [birkhoffAverage, birkhoffSum_apply_sub_birkhoffSum, ← smul_sub]
  /-
    🎉 no goals
  -/

