/-- Two multilinear maps indexed by `Fin n` are equal if they are equal when all arguments are
basis vectors. -/
theorem Basis.ext_multilinear_fin {f g : MultilinearMap R M M₂} {ι₁ : Fin n → Type*}
    (e : ∀ i, Basis (ι₁ i) R (M i))
    (h : ∀ v : ∀ i, ι₁ i, (f fun i => e i (v i)) = g fun i => e i (v i)) : f = g := by
  /-
    R : Type u_1
    n : Nat
    M : Fin n → Type u_3
    M₂ : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : Fin n) → AddCommMonoid (M i)
    inst✝¹ : (i : Fin n) → Module R (M i)
    inst✝ : Module R M₂
    f g : MultilinearMap R M M₂
    ι₁ : Fin n → Type u_6
    e : (i : Fin n) → Basis (ι₁ i) R (M i)
    h : ∀ (v : (i : Fin n) → ι₁ i), Eq (f fun i => (e i) (v i)) (g fun i => (e i)  …
    ⊢ Eq f g
  -/
  induction' n with m hm
    /-
      case zero
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      M : Fin 0 → Type u_3
      inst✝¹ : (i : Fin 0) → AddCommMonoid (M i)
      inst✝ : (i : Fin 0) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin 0 → Type u_6
      e : (i : Fin 0) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin 0) → ι₁ i), Eq (f fun i => (e i) (v i)) (g fun i => (e i)  …
      ⊢ Eq f g
    -/
  · ext x
    /-
      case zero.H
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      M : Fin 0 → Type u_3
      inst✝¹ : (i : Fin 0) → AddCommMonoid (M i)
      inst✝ : (i : Fin 0) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin 0 → Type u_6
      e : (i : Fin 0) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin 0) → ι₁ i), Eq (f fun i => (e i) (v i)) (g fun i => (e i)  …
      x : (i : Fin 0) → M i
      ⊢ Eq (f x) (g x)
    -/
    convert h finZeroElim
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      m : Nat
      hm : ∀ {M : Fin m → Type u_3} [inst : (i : Fin m) → AddCommMonoid (M i)] [inst …
      M : Fin (HAdd.hAdd m 1) → Type u_3
      inst✝¹ : (i : Fin (HAdd.hAdd m 1)) → AddCommMonoid (M i)
      inst✝ : (i : Fin (HAdd.hAdd m 1)) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin (HAdd.hAdd m 1) → Type u_6
      e : (i : Fin (HAdd.hAdd m 1)) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin (HAdd.hAdd m 1)) → ι₁ i), Eq (f fun i => (e i) (v i)) (g f …
      ⊢ Eq f g
    -/
  · apply Function.LeftInverse.injective uncurry_curryLeft
    /-
      case succ.a
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      m : Nat
      hm : ∀ {M : Fin m → Type u_3} [inst : (i : Fin m) → AddCommMonoid (M i)] [inst …
      M : Fin (HAdd.hAdd m 1) → Type u_3
      inst✝¹ : (i : Fin (HAdd.hAdd m 1)) → AddCommMonoid (M i)
      inst✝ : (i : Fin (HAdd.hAdd m 1)) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin (HAdd.hAdd m 1) → Type u_6
      e : (i : Fin (HAdd.hAdd m 1)) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin (HAdd.hAdd m 1)) → ι₁ i), Eq (f fun i => (e i) (v i)) (g f …
      ⊢ Eq f.curryLeft g.curryLeft
    -/
    refine Basis.ext (e 0) ?_
    /-
      case succ.a
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      m : Nat
      hm : ∀ {M : Fin m → Type u_3} [inst : (i : Fin m) → AddCommMonoid (M i)] [inst …
      M : Fin (HAdd.hAdd m 1) → Type u_3
      inst✝¹ : (i : Fin (HAdd.hAdd m 1)) → AddCommMonoid (M i)
      inst✝ : (i : Fin (HAdd.hAdd m 1)) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin (HAdd.hAdd m 1) → Type u_6
      e : (i : Fin (HAdd.hAdd m 1)) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin (HAdd.hAdd m 1)) → ι₁ i), Eq (f fun i => (e i) (v i)) (g f …
      ⊢ ∀ (i : ι₁ 0), Eq (f.curryLeft ((e 0) i)) (g.curryLeft ((e 0) i))
    -/
    intro i
    /-
      case succ.a
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      m : Nat
      hm : ∀ {M : Fin m → Type u_3} [inst : (i : Fin m) → AddCommMonoid (M i)] [inst …
      M : Fin (HAdd.hAdd m 1) → Type u_3
      inst✝¹ : (i : Fin (HAdd.hAdd m 1)) → AddCommMonoid (M i)
      inst✝ : (i : Fin (HAdd.hAdd m 1)) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin (HAdd.hAdd m 1) → Type u_6
      e : (i : Fin (HAdd.hAdd m 1)) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin (HAdd.hAdd m 1)) → ι₁ i), Eq (f fun i => (e i) (v i)) (g f …
      i : ι₁ 0
      ⊢ Eq (f.curryLeft ((e 0) i)) (g.curryLeft ((e 0) i))
    -/
    apply hm (Fin.tail e)
    /-
      case succ.a
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      m : Nat
      hm : ∀ {M : Fin m → Type u_3} [inst : (i : Fin m) → AddCommMonoid (M i)] [inst …
      M : Fin (HAdd.hAdd m 1) → Type u_3
      inst✝¹ : (i : Fin (HAdd.hAdd m 1)) → AddCommMonoid (M i)
      inst✝ : (i : Fin (HAdd.hAdd m 1)) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin (HAdd.hAdd m 1) → Type u_6
      e : (i : Fin (HAdd.hAdd m 1)) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin (HAdd.hAdd m 1)) → ι₁ i), Eq (f fun i => (e i) (v i)) (g f …
      i : ι₁ 0
      ⊢ ∀ (v : (i : Fin m) → ι₁ i.succ), Eq ((f.curryLeft ((e 0) i)) fun i => (Fin.t …
    -/
    intro j
    /-
      case succ.a
      R : Type u_1
      M₂ : Type u_4
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      m : Nat
      hm : ∀ {M : Fin m → Type u_3} [inst : (i : Fin m) → AddCommMonoid (M i)] [inst …
      M : Fin (HAdd.hAdd m 1) → Type u_3
      inst✝¹ : (i : Fin (HAdd.hAdd m 1)) → AddCommMonoid (M i)
      inst✝ : (i : Fin (HAdd.hAdd m 1)) → Module R (M i)
      f g : MultilinearMap R M M₂
      ι₁ : Fin (HAdd.hAdd m 1) → Type u_6
      e : (i : Fin (HAdd.hAdd m 1)) → Basis (ι₁ i) R (M i)
      h : ∀ (v : (i : Fin (HAdd.hAdd m 1)) → ι₁ i), Eq (f fun i => (e i) (v i)) (g f …
      i : ι₁ 0
      j : (i : Fin m) → ι₁ i.succ
      ⊢ Eq ((f.curryLeft ((e 0) i)) fun i => (Fin.tail e i) (j i)) ((g.curryLeft ((e …
    -/
    convert h (Fin.cons i j)
    iterate 2
      rw [curryLeft_apply]
      congr 1 with x
      refine Fin.cases rfl (fun x => ?_) x
      dsimp [Fin.tail]
      rw [Fin.cons_succ, Fin.cons_succ]


/-- Two multilinear maps indexed by a `Fintype` are equal if they are equal when all arguments
are basis vectors. Unlike `Basis.ext_multilinear_fin`, this only uses a single basis; a
dependently-typed version would still be true, but the proof would need a dependently-typed
version of `dom_dom_congr`. -/
theorem Basis.ext_multilinear [Finite ι] {f g : MultilinearMap R (fun _ : ι => M₂) M₃} {ι₁ : Type*}
    (e : Basis ι₁ R M₂) (h : ∀ v : ι → ι₁, (f fun i => e (v i)) = g fun i => e (v i)) : f = g := by
  /-
    R : Type u_1
    ι : Type u_2
    M₂ : Type u_4
    M₃ : Type u_5
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Finite ι
    f g : MultilinearMap R (fun x => M₂) M₃
    ι₁ : Type u_6
    e : Basis ι₁ R M₂
    h : ∀ (v : ι → ι₁), Eq (f fun i => e (v i)) (g fun i => e (v i))
    ⊢ Eq f g
  -/
  cases nonempty_fintype ι
  exact
    (domDomCongr_eq_iff (Fintype.equivFin ι) f g).mp
      (Basis.ext_multilinear_fin (fun _ => e) fun i => h (i ∘ _))

