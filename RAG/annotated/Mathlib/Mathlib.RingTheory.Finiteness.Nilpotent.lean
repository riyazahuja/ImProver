theorem Module.End.isNilpotent_iff_of_finite [Module.Finite R M] {f : End R M} :
    IsNilpotent f ↔ ∀ m : M, ∃ n : ℕ, (f ^ n) m = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    f : Module.End R M
    ⊢ Iff (IsNilpotent f) (∀ (m : M), Exists fun n => Eq ((HPow.hPow f n) m) 0)
  -/
  refine ⟨fun ⟨n, hn⟩ m ↦ ⟨n, by simp [hn]⟩, fun h ↦ ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    f : Module.End R M
    h : ∀ (m : M), Exists fun n => Eq ((HPow.hPow f n) m) 0
    ⊢ IsNilpotent f
  -/
  rcases Module.Finite.out (R := R) (M := M) with ⟨S, hS⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    f : Module.End R M
    h : ∀ (m : M), Exists fun n => Eq ((HPow.hPow f n) m) 0
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    ⊢ IsNilpotent f
  -/
  choose g hg using h
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    f : Module.End R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    g : M → Nat
    hg : ∀ (m : M), Eq ((HPow.hPow f (g m)) m) 0
    ⊢ IsNilpotent f
  -/
  use Finset.sup S g
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    f : Module.End R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    g : M → Nat
    hg : ∀ (m : M), Eq ((HPow.hPow f (g m)) m) 0
    ⊢ Eq (HPow.hPow f (S.sup g)) 0
  -/
  ext m
  /-
    case h.h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    f : Module.End R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    g : M → Nat
    hg : ∀ (m : M), Eq ((HPow.hPow f (g m)) m) 0
    m : M
    ⊢ Eq ((HPow.hPow f (S.sup g)) m) (0 m)
  -/
  have hm : m ∈ Submodule.span R S := by simp [hS]
  induction hm using Submodule.span_induction with
  | mem x hx => exact LinearMap.pow_map_zero_of_le (Finset.le_sup hx) (hg x)
  | zero => simp
  | add => simp_all
  | smul => simp_all


