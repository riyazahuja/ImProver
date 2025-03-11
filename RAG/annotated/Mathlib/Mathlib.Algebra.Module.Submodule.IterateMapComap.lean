/-- The `LinearMap.iterateMapComap f i n K : Submodule R N` is
`f⁻¹(i(⋯(f⁻¹(i(K)))))` (`n` times). -/
def iterateMapComap (n : ℕ) := (fun K : Submodule R N ↦ (K.map i).comap f)^[n]


/-- If `f(K) ≤ i(K)`, then `LinearMap.iterateMapComap` is not decreasing. -/
theorem iterateMapComap_le_succ (K : Submodule R N) (h : K.map f ≤ K.map i) (n : ℕ) :
    f.iterateMapComap i n K ≤ f.iterateMapComap i (n + 1) K := by
  /-
    R : Type u_1
    N : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f i : LinearMap (RingHom.id R) N M
    K : Submodule R N
    h : LE.le (Submodule.map f K) (Submodule.map i K)
    n : Nat
    ⊢ LE.le (f.iterateMapComap i n K) (f.iterateMapComap i (HAdd.hAdd n 1) K)
  -/
  nth_rw 2 [iterateMapComap]
  /-
    R : Type u_1
    N : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f i : LinearMap (RingHom.id R) N M
    K : Submodule R N
    h : LE.le (Submodule.map f K) (Submodule.map i K)
    n : Nat
    ⊢ LE.le (f.iterateMapComap i n K) (Nat.iterate (fun K => Submodule.comap f (Su …
  -/
  rw [iterate_succ', Function.comp_apply, ← iterateMapComap, ← map_le_iff_le_comap]
  induction n with
  | zero => exact h
  | succ n ih =>
    simp_rw [iterateMapComap, iterate_succ', Function.comp_apply]
    calc
      _ ≤ (f.iterateMapComap i n K).map i := map_comap_le _ _
      _ ≤ (((f.iterateMapComap i n K).map f).comap f).map i := map_mono (le_comap_map _ _)
      _ ≤ _ := map_mono (comap_mono ih)


/-- If `f` is surjective, `i` is injective, and there exists some `m` such that
`LinearMap.iterateMapComap f i m K = LinearMap.iterateMapComap f i (m + 1) K`,
then for any `n`,
`LinearMap.iterateMapComap f i n K = LinearMap.iterateMapComap f i (n + 1) K`.
In particular, by taking `n = 0`, the kernel of `f` is contained in `K`
(`LinearMap.ker_le_of_iterateMapComap_eq_succ`),
which is a consequence of `LinearMap.ker_le_comap`. -/
theorem iterateMapComap_eq_succ (K : Submodule R N)
    (m : ℕ) (heq : f.iterateMapComap i m K = f.iterateMapComap i (m + 1) K)
    (hf : Surjective f) (hi : Injective i) (n : ℕ) :
    f.iterateMapComap i n K = f.iterateMapComap i (n + 1) K := by
  induction n with
  | zero =>
    contrapose! heq
    induction m with
    | zero => exact heq
    | succ m ih =>
      rw [iterateMapComap, iterateMapComap, iterate_succ', iterate_succ']
      exact fun H ↦ ih (map_injective_of_injective hi (comap_injective_of_surjective hf H))
  | succ n ih =>
    rw [iterateMapComap, iterateMapComap, iterate_succ', iterate_succ',
      Function.comp_apply, Function.comp_apply, ← iterateMapComap, ← iterateMapComap, ih]


/-- If `f` is surjective, `i` is injective, and there exists some `m` such that
`LinearMap.iterateMapComap f i m K = LinearMap.iterateMapComap f i (m + 1) K`,
then the kernel of `f` is contained in `K`.
This is a corollary of `LinearMap.iterateMapComap_eq_succ` and `LinearMap.ker_le_comap`.
As a special case, if one can take `K` to be zero,
then `f` is injective. This is the key result for establishing the strong rank condition
for noetherian rings. -/
theorem ker_le_of_iterateMapComap_eq_succ (K : Submodule R N)
    (m : ℕ) (heq : f.iterateMapComap i m K = f.iterateMapComap i (m + 1) K)
    (hf : Surjective f) (hi : Injective i) : LinearMap.ker f ≤ K := by
  /-
    R : Type u_1
    N : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f i : LinearMap (RingHom.id R) N M
    K : Submodule R N
    m : Nat
    heq : Eq (f.iterateMapComap i m K) (f.iterateMapComap i (HAdd.hAdd m 1) K)
    hf : Function.Surjective ⇑f
    hi : Function.Injective ⇑i
    ⊢ LE.le (LinearMap.ker f) K
  -/
  rw [show K = _ from f.iterateMapComap_eq_succ i K m heq hf hi 0]
  /-
    R : Type u_1
    N : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f i : LinearMap (RingHom.id R) N M
    K : Submodule R N
    m : Nat
    heq : Eq (f.iterateMapComap i m K) (f.iterateMapComap i (HAdd.hAdd m 1) K)
    hf : Function.Surjective ⇑f
    hi : Function.Injective ⇑i
    ⊢ LE.le (LinearMap.ker f) (f.iterateMapComap i (HAdd.hAdd 0 1) K)
  -/
  exact f.ker_le_comap
  /-
    🎉 no goals
  -/


