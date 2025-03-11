/-- Two multilinear maps from finite families are equal if they agree on the generators.

This is a multilinear version of `LinearMap.pi_ext`. -/
@[ext]
theorem pi_ext [Finite ι] [∀ i, Finite (κ i)] [∀ i, DecidableEq (κ i)]
    ⦃f g : MultilinearMap R (fun i ↦ Π j : κ i, M i j) N⦄
    (h : ∀ p : Π i, κ i,
      f.compLinearMap (fun i => LinearMap.single R _ (p i)) =
      g.compLinearMap (fun i => LinearMap.single R _ (p i))) : f = g := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    ⊢ Eq f g
  -/
  ext x
  /-
    case H
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    ⊢ Eq (f x) (g x)
  -/
  show f (fun i ↦ x i) = g (fun i ↦ x i)
  /-
    case H
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    ⊢ Eq (f fun i => x i) (g fun i => x i)
  -/
  obtain ⟨i⟩ := nonempty_fintype ι
  /-
    case H.intro
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    ⊢ Eq (f fun i => x i) (g fun i => x i)
  -/
  have (i) := (nonempty_fintype (κ i)).some
  /-
    case H.intro
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    this : (i : ι) → Fintype (κ i)
    ⊢ Eq (f fun i => x i) (g fun i => x i)
  -/
  have := Classical.decEq ι
  /-
    case H.intro
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    this✝ : (i : ι) → Fintype (κ i)
    this : DecidableEq ι
    ⊢ Eq (f fun i => x i) (g fun i => x i)
  -/
  rw [funext (fun i ↦ Eq.symm (Finset.univ_sum_single (x i)))]
  /-
    case H.intro
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    this✝ : (i : ι) → Fintype (κ i)
    this : DecidableEq ι
    ⊢ Eq (f fun i => (fun i => Finset.univ.sum fun i_1 => Pi.single i_1 (x i i_1)) …
  -/
  simp_rw [MultilinearMap.map_sum_finset]
  /-
    case H.intro
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    this✝ : (i : ι) → Fintype (κ i)
    this : DecidableEq ι
    ⊢ Eq ((Fintype.piFinset fun i => Finset.univ).sum fun r => f fun i => Pi.singl …
  -/
  congr! 1 with p
  /-
    case H.intro.a
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => LinearMap.single R (M  …
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    this✝ : (i : ι) → Fintype (κ i)
    this : DecidableEq ι
    p : (a : ι) → κ a
    a✝ : Membership.mem (Fintype.piFinset fun i => Finset.univ) p
    ⊢ Eq (f fun i => Pi.single (p i) (x i (p i))) (g fun i => Pi.single (p i) (x i …
  -/
  simp_rw [MultilinearMap.ext_iff] at h
  /-
    case H.intro.a
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : Module R N
    inst✝² : Finite ι
    inst✝¹ : ∀ (i : ι), Finite (κ i)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => (j : κ i) → M i j) N
    x : (i : ι) → (j : κ i) → M i j
    i : Fintype ι
    this✝ : (i : ι) → Fintype (κ i)
    this : DecidableEq ι
    p : (a : ι) → κ a
    a✝ : Membership.mem (Fintype.piFinset fun i => Finset.univ) p
    h : ∀ (p : (i : ι) → κ i) (x : (i : ι) → M i (p i)), Eq ((f.compLinearMap fun  …
    ⊢ Eq (f fun i => Pi.single (p i) (x i (p i))) (g fun i => Pi.single (p i) (x i …
  -/
  exact h _ _
  /-
    🎉 no goals
  -/


/--
Given a family of indices `κ` and a multilinear map `f p` for each way `p` to select one index from
each family, `piFamily f` maps a family of functions (one for each domain `κ i`) into a function
from each selection of indices (with domain `Π i, κ i`).
-/
@[simps]
def piFamily (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    MultilinearMap R (fun i => Π j : κ i, M i j) (Π t : Π i, κ i, N t) where
  toFun x := fun p => f p (fun i => x i (p i))
  map_update_add' {dec} m i x y := funext fun p => by
    /-
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → (j : κ i) → M i j
      i : ι
      x y : (j : κ i) → M i j
      p : (i : ι) → κ i
      ⊢ Eq ((fun x p => (f p) fun i => x i (p i)) (Function.update m i (HAdd.hAdd x  …
    -/
    cases Subsingleton.elim dec (by infer_instance)
    /-
      case refl
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → (j : κ i) → M i j
      i : ι
      x y : (j : κ i) → M i j
      p : (i : ι) → κ i
      ⊢ Eq ((fun x p => (f p) fun i => x i (p i)) (Function.update m i (HAdd.hAdd x  …
    -/
    dsimp
    /-
      case refl
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → (j : κ i) → M i j
      i : ι
      x y : (j : κ i) → M i j
      p : (i : ι) → κ i
      ⊢ Eq ((f p) fun i_1 => Function.update m i (HAdd.hAdd x y) i_1 (p i_1)) (HAdd. …
    -/
    simp_rw [Function.apply_update (fun i m => m (p i)) m, Pi.add_apply, (f p).map_update_add]
    /-
      🎉 no goals
    -/
  map_update_smul' {dec} m i c x := funext fun p => by
    /-
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → (j : κ i) → M i j
      i : ι
      c : R
      x : (j : κ i) → M i j
      p : (i : ι) → κ i
      ⊢ Eq ((fun x p => (f p) fun i => x i (p i)) (Function.update m i (HSMul.hSMul  …
    -/
    cases Subsingleton.elim dec (by infer_instance)
    /-
      case refl
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → (j : κ i) → M i j
      i : ι
      c : R
      x : (j : κ i) → M i j
      p : (i : ι) → κ i
      ⊢ Eq ((fun x p => (f p) fun i => x i (p i)) (Function.update m i (HSMul.hSMul  …
    -/
    dsimp
    /-
      case refl
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → (j : κ i) → M i j
      i : ι
      c : R
      x : (j : κ i) → M i j
      p : (i : ι) → κ i
      ⊢ Eq ((f p) fun i_1 => Function.update m i (HSMul.hSMul c x) i_1 (p i_1)) (HSM …
    -/
    simp_rw [Function.apply_update (fun i m => m (p i)) m, Pi.smul_apply, (f p).map_update_smul]
    /-
      🎉 no goals
    -/


/-- When applied to a family of finitely-supported functions each supported on a single element,
`piFamily` is itself supported on a single element, with value equal to the map `f` applied
at that point. -/
@[simp]
theorem piFamily_single [Fintype ι] [∀ i, DecidableEq (κ i)]
    (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p))
    (p : ∀ i, κ i) (m : ∀ i, M i (p i)) :
    piFamily f (fun i => Pi.single (p i) (m i)) = Pi.single p (f p m) := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝² : (p : (i : ι) → κ i) → Module R (N p)
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (κ i)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    p : (i : ι) → κ i
    m : (i : ι) → M i (p i)
    ⊢ Eq ((MultilinearMap.piFamily f) fun i => Pi.single (p i) (m i)) (Pi.single p …
  -/
  ext q
  /-
    case h
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝² : (p : (i : ι) → κ i) → Module R (N p)
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (κ i)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    p : (i : ι) → κ i
    m : (i : ι) → M i (p i)
    q : (i : ι) → κ i
    ⊢ Eq ((MultilinearMap.piFamily f) (fun i => Pi.single (p i) (m i)) q) (Pi.sing …
  -/
  obtain rfl | hpq := eq_or_ne p q
    /-
      case h.inl
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      ⊢ Eq ((MultilinearMap.piFamily f) (fun i => Pi.single (p i) (m i)) p) (Pi.sing …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      hpq : Ne p q
      ⊢ Eq ((MultilinearMap.piFamily f) (fun i => Pi.single (p i) (m i)) q) (Pi.sing …
    -/
  · rw [Pi.single_eq_of_ne' hpq]
    /-
      case h.inr
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      hpq : Ne p q
      ⊢ Eq ((MultilinearMap.piFamily f) (fun i => Pi.single (p i) (m i)) q) 0
    -/
    rw [Function.ne_iff] at hpq
    /-
      case h.inr
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      hpq : Exists fun a => Ne (p a) (q a)
      ⊢ Eq ((MultilinearMap.piFamily f) (fun i => Pi.single (p i) (m i)) q) 0
    -/
    obtain ⟨i, hpqi⟩ := hpq
    /-
      case h.inr.intro
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      i : ι
      hpqi : Ne (p i) (q i)
      ⊢ Eq ((MultilinearMap.piFamily f) (fun i => Pi.single (p i) (m i)) q) 0
    -/
    apply (f q).map_coord_zero i
    /-
      case h.inr.intro
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      i : ι
      hpqi : Ne (p i) (q i)
      ⊢ Eq ((fun i => Pi.single (p i) (m i)) i (q i)) 0
    -/
    simp_rw [Pi.single_eq_of_ne' hpqi]
    /-
      🎉 no goals
    -/


/-- When only one member of the family of multilinear maps is nonzero, the result consists only of
the component from that member. -/
@[simp]
theorem piFamily_single_left_apply [Fintype ι] [∀ i, DecidableEq (κ i)]
    (p : Π i, κ i) (f : MultilinearMap R (fun i ↦ M i (p i)) (N p)) (x : Π i j, M i j) :
    piFamily (Pi.single p f) x = Pi.single p (f fun i => x i (p i)) := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝² : (p : (i : ι) → κ i) → Module R (N p)
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (κ i)
    p : (i : ι) → κ i
    f : MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → (j : κ i) → M i j
    ⊢ Eq ((MultilinearMap.piFamily (Pi.single p f)) x) (Pi.single p (f fun i => x  …
  -/
  ext p'
  /-
    case h
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝² : (p : (i : ι) → κ i) → Module R (N p)
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → DecidableEq (κ i)
    p : (i : ι) → κ i
    f : MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → (j : κ i) → M i j
    p' : (i : ι) → κ i
    ⊢ Eq ((MultilinearMap.piFamily (Pi.single p f)) x p') (Pi.single p (f fun i => …
  -/
  obtain rfl | hp := eq_or_ne p p'
    /-
      case h.inl
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      p : (i : ι) → κ i
      f : MultilinearMap R (fun i => M i (p i)) (N p)
      x : (i : ι) → (j : κ i) → M i j
      ⊢ Eq ((MultilinearMap.piFamily (Pi.single p f)) x p) (Pi.single p (f fun i =>  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝⁴ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝³ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝² : (p : (i : ι) → κ i) → Module R (N p)
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → DecidableEq (κ i)
      p : (i : ι) → κ i
      f : MultilinearMap R (fun i => M i (p i)) (N p)
      x : (i : ι) → (j : κ i) → M i j
      p' : (i : ι) → κ i
      hp : Ne p p'
      ⊢ Eq ((MultilinearMap.piFamily (Pi.single p f)) x p') (Pi.single p (f fun i => …
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/


theorem piFamily_single_left [Fintype ι] [∀ i, DecidableEq (κ i)]
    (p : Π i, κ i) (f : MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    piFamily (Pi.single p f) =
      (LinearMap.single R _ p).compMultilinearMap (f.compLinearMap fun i => .proj (p i)) :=
  ext <| piFamily_single_left_apply _ _


@[simp]
theorem piFamily_compLinearMap_lsingle [Fintype ι] [∀ i, DecidableEq (κ i)]
    (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) (p : ∀ i, κ i) :
    (piFamily f).compLinearMap (fun i => LinearMap.single _ _ (p i))
      = (LinearMap.single _ _ p).compMultilinearMap (f p) :=
  MultilinearMap.ext <| piFamily_single f p


@[simp]
theorem piFamily_zero :
    piFamily (0 : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) = 0 := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁴ : Semiring R
    inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝ : (p : (i : ι) → κ i) → Module R (N p)
    ⊢ Eq (MultilinearMap.piFamily 0) 0
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem piFamily_add (f g : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    piFamily (f + g) = piFamily f + piFamily g := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁴ : Semiring R
    inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝ : (p : (i : ι) → κ i) → Module R (N p)
    f g : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    ⊢ Eq (MultilinearMap.piFamily (HAdd.hAdd f g)) (HAdd.hAdd (MultilinearMap.piFa …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem piFamily_smul
    [Monoid S] [∀ p, DistribMulAction S (N p)] [∀ p, SMulCommClass R S (N p)]
    (s : S) (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    piFamily (s • f) = s • piFamily f := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    S : Type uS
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝² : Monoid S
    inst✝¹ : (p : (i : ι) → κ i) → DistribMulAction S (N p)
    inst✝ : ∀ (p : (i : ι) → κ i), SMulCommClass R S (N p)
    s : S
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    ⊢ Eq (MultilinearMap.piFamily (HSMul.hSMul s f)) (HSMul.hSMul s (MultilinearMa …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- `MultilinearMap.piFamily` as a linear map. -/
@[simps]
def piFamilyₗ :
    (Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p))
      →ₗ[R] MultilinearMap R (fun i => Π j : κ i, M i j) (Π t : Π i, κ i, N t) where
  toFun := piFamily
  map_add' := piFamily_add
  map_smul' := piFamily_smul


