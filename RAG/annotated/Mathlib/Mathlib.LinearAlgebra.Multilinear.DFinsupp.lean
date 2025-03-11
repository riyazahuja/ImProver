/-- Two multilinear maps from finitely supported functions are equal if they agree on the
generators.

This is a multilinear version of `DFinsupp.lhom_ext'`. -/
@[ext]
theorem dfinsupp_ext [∀ i, DecidableEq (κ i)]
    ⦃f g : MultilinearMap R (fun i ↦ Π₀ j : κ i, M i j) N⦄
    (h : ∀ p : Π i, κ i,
      f.compLinearMap (fun i => DFinsupp.lsingle (p i)) =
      g.compLinearMap (fun i => DFinsupp.lsingle (p i))) : f = g := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : Type uN
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝³ : AddCommMonoid N
    inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝¹ : Module R N
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => DFinsupp fun j => M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => DFinsupp.lsingle (p i) …
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
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝³ : AddCommMonoid N
    inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝¹ : Module R N
    inst✝ : (i : ι) → DecidableEq (κ i)
    f g : MultilinearMap R (fun i => DFinsupp fun j => M i j) N
    h : ∀ (p : (i : ι) → κ i), Eq (f.compLinearMap fun i => DFinsupp.lsingle (p i) …
    x : (i : ι) → DFinsupp fun j => M i j
    ⊢ Eq (f x) (g x)
  -/
  show f (fun i ↦ x i) = g (fun i ↦ x i)
  classical
  rw [funext (fun i ↦ Eq.symm (DFinsupp.sum_single (f := x i)))]
  simp_rw [DFinsupp.sum, MultilinearMap.map_sum_finset]
  congr! 1 with p
  simp_rw [MultilinearMap.ext_iff] at h
  exact h _ _


/--
Given a family of indices `κ` and a multilinear map `f p` for each way `p` to select one index from
each family, `dfinsuppFamily f` maps a family of finitely-supported functions (one for each domain
`κ i`) into a finitely-supported function from each selection of indices (with domain `Π i, κ i`).

Strictly this doesn't need multilinearity, only the fact that `f p m = 0` whenever `m i = 0` for
some `i`.

This is the `DFinsupp` version of `MultilinearMap.piFamily`.
-/
@[simps]
def dfinsuppFamily
    (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    MultilinearMap R (fun i => Π₀ j : κ i, M i j) (Π₀ t : Π i, κ i, N t) where
  toFun x :=
  { toFun := fun p => f p (fun i => x i (p i))
    support' := (Trunc.finChoice fun i => (x i).support').map fun s => ⟨
      Finset.univ.val.pi (fun i ↦ (s i).val) |>.map fun f i => f i (Finset.mem_univ _),
      fun p => by
        simp only [Multiset.mem_map, Multiset.mem_pi, Finset.mem_val, Finset.mem_univ,
          forall_true_left]
        /-
          ι : Type uι
          κ : ι → Type uκ
          S : Type uS
          R : Type uR
          M : (i : ι) → κ i → Type uM
          N : ((i : ι) → κ i) → Type uN
          inst✝⁶ : DecidableEq ι
          inst✝⁵ : Fintype ι
          inst✝⁴ : Semiring R
          inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
          inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
          inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
          inst✝ : (p : (i : ι) → κ i) → Module R (N p)
          f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
          x : (i : ι) → DFinsupp fun j => M i j
          s : (i : ι) → Subtype fun s => ∀ (i_1 : κ i), Or (Membership.mem s i_1) (Eq (( …
          p : (i : ι) → κ i
          ⊢ Or (Exists fun a => And (∀ (a_1 : ι), Membership.mem (↑(s a_1)) (a a_1 ⋯)) ( …
        -/
        simp_rw [or_iff_not_imp_right]
        /-
          ι : Type uι
          κ : ι → Type uκ
          S : Type uS
          R : Type uR
          M : (i : ι) → κ i → Type uM
          N : ((i : ι) → κ i) → Type uN
          inst✝⁶ : DecidableEq ι
          inst✝⁵ : Fintype ι
          inst✝⁴ : Semiring R
          inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
          inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
          inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
          inst✝ : (p : (i : ι) → κ i) → Module R (N p)
          f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
          x : (i : ι) → DFinsupp fun j => M i j
          s : (i : ι) → Subtype fun s => ∀ (i_1 : κ i), Or (Membership.mem s i_1) (Eq (( …
          p : (i : ι) → κ i
          ⊢ Not (Eq ((f p) fun i => (x i) (p i)) 0) → Exists fun a => And (∀ (a_1 : ι),  …
        -/
        intro h
        /-
          ι : Type uι
          κ : ι → Type uκ
          S : Type uS
          R : Type uR
          M : (i : ι) → κ i → Type uM
          N : ((i : ι) → κ i) → Type uN
          inst✝⁶ : DecidableEq ι
          inst✝⁵ : Fintype ι
          inst✝⁴ : Semiring R
          inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
          inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
          inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
          inst✝ : (p : (i : ι) → κ i) → Module R (N p)
          f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
          x : (i : ι) → DFinsupp fun j => M i j
          s : (i : ι) → Subtype fun s => ∀ (i_1 : κ i), Or (Membership.mem s i_1) (Eq (( …
          p : (i : ι) → κ i
          h : Not (Eq ((f p) fun i => (x i) (p i)) 0)
          ⊢ Exists fun a => And (∀ (a_1 : ι), Membership.mem (↑(s a_1)) (a a_1 ⋯)) (Eq ( …
        -/
        push_neg at h
        /-
          ι : Type uι
          κ : ι → Type uκ
          S : Type uS
          R : Type uR
          M : (i : ι) → κ i → Type uM
          N : ((i : ι) → κ i) → Type uN
          inst✝⁶ : DecidableEq ι
          inst✝⁵ : Fintype ι
          inst✝⁴ : Semiring R
          inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
          inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
          inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
          inst✝ : (p : (i : ι) → κ i) → Module R (N p)
          f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
          x : (i : ι) → DFinsupp fun j => M i j
          s : (i : ι) → Subtype fun s => ∀ (i_1 : κ i), Or (Membership.mem s i_1) (Eq (( …
          p : (i : ι) → κ i
          h : Ne ((f p) fun i => (x i) (p i)) 0
          ⊢ Exists fun a => And (∀ (a_1 : ι), Membership.mem (↑(s a_1)) (a a_1 ⋯)) (Eq ( …
        -/
        refine ⟨fun i _ => p i, fun i => (s i).prop _ |>.resolve_right ?_, rfl⟩
        /-
          ι : Type uι
          κ : ι → Type uκ
          S : Type uS
          R : Type uR
          M : (i : ι) → κ i → Type uM
          N : ((i : ι) → κ i) → Type uN
          inst✝⁶ : DecidableEq ι
          inst✝⁵ : Fintype ι
          inst✝⁴ : Semiring R
          inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
          inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
          inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
          inst✝ : (p : (i : ι) → κ i) → Module R (N p)
          f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
          x : (i : ι) → DFinsupp fun j => M i j
          s : (i : ι) → Subtype fun s => ∀ (i_1 : κ i), Or (Membership.mem s i_1) (Eq (( …
          p : (i : ι) → κ i
          h : Ne ((f p) fun i => (x i) (p i)) 0
          i : ι
          ⊢ Not (Eq ((x i).toFun ((fun i x => p i) i ⋯)) 0)
        -/
        exact mt ((f p).map_coord_zero (m := fun i => x i _) i) h⟩}
        /-
          🎉 no goals
        -/
  map_update_add' {dec} m i x y := DFinsupp.ext fun p => by
    /-
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Fintype ι
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → DFinsupp fun j => M i j
      i : ι
      x y : DFinsupp fun j => M i j
      p : (i : ι) → κ i
      ⊢ Eq (((fun x => { toFun := fun p => (f p) fun i => (x i) (p i), support' := T …
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
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Fintype ι
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → DFinsupp fun j => M i j
      i : ι
      x y : DFinsupp fun j => M i j
      p : (i : ι) → κ i
      ⊢ Eq (((fun x => { toFun := fun p => (f p) fun i => (x i) (p i), support' := T …
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
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Fintype ι
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → DFinsupp fun j => M i j
      i : ι
      x y : DFinsupp fun j => M i j
      p : (i : ι) → κ i
      ⊢ Eq ((f p) fun i_1 => (Function.update m i (HAdd.hAdd x y) i_1) (p i_1)) (HAd …
    -/
    simp_rw [Function.apply_update (fun i m => m (p i)) m, DFinsupp.add_apply, (f p).map_update_add]
    /-
      🎉 no goals
    -/
  map_update_smul' {dec} m i c x := DFinsupp.ext fun p => by
    /-
      ι : Type uι
      κ : ι → Type uκ
      S : Type uS
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Fintype ι
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → DFinsupp fun j => M i j
      i : ι
      c : R
      x : DFinsupp fun j => M i j
      p : (i : ι) → κ i
      ⊢ Eq (((fun x => { toFun := fun p => (f p) fun i => (x i) (p i), support' := T …
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
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Fintype ι
      inst✝⁴ : Semiring R
      inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝ : (p : (i : ι) → κ i) → Module R (N p)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      dec : DecidableEq ι
      m : (i : ι) → DFinsupp fun j => M i j
      i : ι
      c : R
      x : DFinsupp fun j => M i j
      p : (i : ι) → κ i
      ⊢ Eq (((fun x => { toFun := fun p => (f p) fun i => (x i) (p i), support' := T …
    -/
    dsimp
    simp_rw [Function.apply_update (fun i m => m (p i)) m, DFinsupp.smul_apply,
      (f p).map_update_smul]


theorem support_dfinsuppFamily_subset
    [∀ i, DecidableEq (κ i)]
    [∀ i j, (x : M i j) → Decidable (x ≠ 0)] [∀ i, (x : N i) → Decidable (x ≠ 0)]
    (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p))
    (x : ∀ i, Π₀ j : κ i, M i j) :
    (dfinsuppFamily f x).support ⊆ Fintype.piFinset fun i => (x i).support := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Fintype ι
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝² : (i : ι) → DecidableEq (κ i)
    inst✝¹ : (i : ι) → (j : κ i) → (x : M i j) → Decidable (Ne x 0)
    inst✝ : (i : (i : ι) → κ i) → (x : N i) → Decidable (Ne x 0)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → DFinsupp fun j => M i j
    ⊢ HasSubset.Subset ((MultilinearMap.dfinsuppFamily f) x).support (Fintype.piFi …
  -/
  intro p hp
  simp only [DFinsupp.mem_support_toFun, dfinsuppFamily_apply_toFun, ne_eq,
    Fintype.mem_piFinset] at hp ⊢
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Fintype ι
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝² : (i : ι) → DecidableEq (κ i)
    inst✝¹ : (i : ι) → (j : κ i) → (x : M i j) → Decidable (Ne x 0)
    inst✝ : (i : (i : ι) → κ i) → (x : N i) → Decidable (Ne x 0)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → DFinsupp fun j => M i j
    p : (i : ι) → κ i
    hp : Not (Eq ((f p) fun i => (x i) (p i)) 0)
    ⊢ ∀ (a : ι), Not (Eq ((x a) (p a)) 0)
  -/
  intro i
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Fintype ι
    inst✝⁷ : Semiring R
    inst✝⁶ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝⁵ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝⁴ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝³ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝² : (i : ι) → DecidableEq (κ i)
    inst✝¹ : (i : ι) → (j : κ i) → (x : M i j) → Decidable (Ne x 0)
    inst✝ : (i : (i : ι) → κ i) → (x : N i) → Decidable (Ne x 0)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → DFinsupp fun j => M i j
    p : (i : ι) → κ i
    hp : Not (Eq ((f p) fun i => (x i) (p i)) 0)
    i : ι
    ⊢ Not (Eq ((x i) (p i)) 0)
  -/
  exact mt ((f p).map_coord_zero (m := fun i => x i _) i) hp
  /-
    🎉 no goals
  -/


/-- When applied to a family of finitely-supported functions each supported on a single element,
`dfinsuppFamily` is itself supported on a single element, with value equal to the map `f` applied
at that point. -/
@[simp]
theorem dfinsuppFamily_single [∀ i, DecidableEq (κ i)]
    (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p))
    (p : ∀ i, κ i) (m : ∀ i, M i (p i)) :
    dfinsuppFamily f (fun i => .single (p i) (m i)) = DFinsupp.single p (f p m) := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    p : (i : ι) → κ i
    m : (i : ι) → M i (p i)
    ⊢ Eq ((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i))  …
  -/
  ext q
  /-
    case h
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝ : (i : ι) → DecidableEq (κ i)
    f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    p : (i : ι) → κ i
    m : (i : ι) → M i (p i)
    q : (i : ι) → κ i
    ⊢ Eq (((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i)) …
  -/
  obtain rfl | hpq := eq_or_ne p q
    /-
      case h.inl
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      ⊢ Eq (((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i)) …
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
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      hpq : Ne p q
      ⊢ Eq (((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i)) …
    -/
  · rw [DFinsupp.single_eq_of_ne hpq]
    /-
      case h.inr
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      hpq : Ne p q
      ⊢ Eq (((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i)) …
    -/
    rw [Function.ne_iff] at hpq
    /-
      case h.inr
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      hpq : Exists fun a => Ne (p a) (q a)
      ⊢ Eq (((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i)) …
    -/
    obtain ⟨i, hpqi⟩ := hpq
    /-
      case h.inr.intro
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      i : ι
      hpqi : Ne (p i) (q i)
      ⊢ Eq (((MultilinearMap.dfinsuppFamily f) fun i => DFinsupp.single (p i) (m i)) …
    -/
    apply (f q).map_coord_zero i
    /-
      case h.inr.intro
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      f : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
      p : (i : ι) → κ i
      m : (i : ι) → M i (p i)
      q : (i : ι) → κ i
      i : ι
      hpqi : Ne (p i) (q i)
      ⊢ Eq (((fun i => DFinsupp.single (p i) (m i)) i) (q i)) 0
    -/
    simp_rw [DFinsupp.single_eq_of_ne hpqi]
    /-
      🎉 no goals
    -/


/-- When only one member of the family of multilinear maps is nonzero, the result consists only of
the component from that member. -/
@[simp]
theorem dfinsuppFamily_single_left_apply [∀ i, DecidableEq (κ i)]
    (p : Π i, κ i) (f : MultilinearMap R (fun i ↦ M i (p i)) (N p)) (x : Π i, Π₀ j, M i j) :
    dfinsuppFamily (Pi.single p f) x = DFinsupp.single p (f fun i => x _ (p i)) := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝ : (i : ι) → DecidableEq (κ i)
    p : (i : ι) → κ i
    f : MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → DFinsupp fun j => M i j
    ⊢ Eq ((MultilinearMap.dfinsuppFamily (Pi.single p f)) x) (DFinsupp.single p (f …
  -/
  ext p'
  /-
    case h
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
    inst✝ : (i : ι) → DecidableEq (κ i)
    p : (i : ι) → κ i
    f : MultilinearMap R (fun i => M i (p i)) (N p)
    x : (i : ι) → DFinsupp fun j => M i j
    p' : (i : ι) → κ i
    ⊢ Eq (((MultilinearMap.dfinsuppFamily (Pi.single p f)) x) p') ((DFinsupp.singl …
  -/
  obtain rfl | hp := eq_or_ne p p'
    /-
      case h.inl
      ι : Type uι
      κ : ι → Type uκ
      R : Type uR
      M : (i : ι) → κ i → Type uM
      N : ((i : ι) → κ i) → Type uN
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      p : (i : ι) → κ i
      f : MultilinearMap R (fun i => M i (p i)) (N p)
      x : (i : ι) → DFinsupp fun j => M i j
      ⊢ Eq (((MultilinearMap.dfinsuppFamily (Pi.single p f)) x) p) ((DFinsupp.single …
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
      inst✝⁷ : DecidableEq ι
      inst✝⁶ : Fintype ι
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
      inst✝³ : (p : (i : ι) → κ i) → AddCommMonoid (N p)
      inst✝² : (i : ι) → (k : κ i) → Module R (M i k)
      inst✝¹ : (p : (i : ι) → κ i) → Module R (N p)
      inst✝ : (i : ι) → DecidableEq (κ i)
      p : (i : ι) → κ i
      f : MultilinearMap R (fun i => M i (p i)) (N p)
      x : (i : ι) → DFinsupp fun j => M i j
      p' : (i : ι) → κ i
      hp : Ne p p'
      ⊢ Eq (((MultilinearMap.dfinsuppFamily (Pi.single p f)) x) p') ((DFinsupp.singl …
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/


theorem dfinsuppFamily_single_left [∀ i, DecidableEq (κ i)]
    (p : Π i, κ i) (f : MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    dfinsuppFamily (Pi.single p f) =
      (DFinsupp.lsingle p).compMultilinearMap (f.compLinearMap fun i => DFinsupp.lapply (p i)) :=
  ext <| dfinsuppFamily_single_left_apply _ _


@[simp]
theorem dfinsuppFamily_compLinearMap_lsingle [∀ i, DecidableEq (κ i)]
    (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) (p : ∀ i, κ i) :
    (dfinsuppFamily f).compLinearMap (fun i => DFinsupp.lsingle (p i))
      = (DFinsupp.lsingle p).compMultilinearMap (f p) :=
  MultilinearMap.ext <| dfinsuppFamily_single f p


@[simp]
theorem dfinsuppFamily_zero :
    dfinsuppFamily (0 : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) = 0 := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Fintype ι
    inst✝⁴ : Semiring R
    inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝ : (p : (i : ι) → κ i) → Module R (N p)
    ⊢ Eq (MultilinearMap.dfinsuppFamily 0) 0
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem dfinsuppFamily_add (f g : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    dfinsuppFamily (f + g) = dfinsuppFamily f + dfinsuppFamily g := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Fintype ι
    inst✝⁴ : Semiring R
    inst✝³ : (i : ι) → (k : κ i) → AddCommMonoid (M i k)
    inst✝² : (p : (i : ι) → κ i) → AddCommMonoid (N p)
    inst✝¹ : (i : ι) → (k : κ i) → Module R (M i k)
    inst✝ : (p : (i : ι) → κ i) → Module R (N p)
    f g : (p : (i : ι) → κ i) → MultilinearMap R (fun i => M i (p i)) (N p)
    ⊢ Eq (MultilinearMap.dfinsuppFamily (HAdd.hAdd f g)) (HAdd.hAdd (MultilinearMa …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem dfinsuppFamily_smul
    [Monoid S] [∀ p, DistribMulAction S (N p)] [∀ p, SMulCommClass R S (N p)]
    (s : S) (f : Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p)) :
    dfinsuppFamily (s • f) = s • dfinsuppFamily f := by
  /-
    ι : Type uι
    κ : ι → Type uκ
    S : Type uS
    R : Type uR
    M : (i : ι) → κ i → Type uM
    N : ((i : ι) → κ i) → Type uN
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Fintype ι
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
    ⊢ Eq (MultilinearMap.dfinsuppFamily (HSMul.hSMul s f)) (HSMul.hSMul s (Multili …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- `MultilinearMap.dfinsuppFamily` as a linear map. -/
@[simps]
def dfinsuppFamilyₗ :
    (Π (p : Π i, κ i), MultilinearMap R (fun i ↦ M i (p i)) (N p))
      →ₗ[R] MultilinearMap R (fun i => Π₀ j : κ i, M i j) (Π₀ t : Π i, κ i, N t) where
  toFun := dfinsuppFamily
  map_add' := dfinsuppFamily_add
  map_smul' := dfinsuppFamily_smul


