/-- `A.IsTotallyUnimodular` means that every square submatrix of `A` (not necessarily contiguous)
has determinant `0` or `1` or `-1`; that is, the determinant is in the range of `SignType.cast`. -/
def IsTotallyUnimodular (A : Matrix m n R) : Prop :=
  ∀ k : ℕ, ∀ f : Fin k → m, ∀ g : Fin k → n, f.Injective → g.Injective →
    (A.submatrix f g).det ∈ Set.range SignType.cast


lemma isTotallyUnimodular_iff (A : Matrix m n R) : A.IsTotallyUnimodular ↔
    ∀ k : ℕ, ∀ f : Fin k → m, ∀ g : Fin k → n,
      (A.submatrix f g).det ∈ Set.range SignType.cast := by
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    ⊢ Iff A.IsTotallyUnimodular (∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Memb …
  -/
  constructor <;> intro hA
    /-
      case mp
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      hA : A.IsTotallyUnimodular
      ⊢ ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range SignT …
    -/
  · intro k f g
    /-
      case mp
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      hA : A.IsTotallyUnimodular
      k : Nat
      f : Fin k → m
      g : Fin k → n
      ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
    -/
    by_cases hfg : f.Injective ∧ g.Injective
      /-
        case pos
        m : Type u_1
        n : Type u_3
        R : Type u_5
        inst✝ : CommRing R
        A : Matrix m n R
        hA : A.IsTotallyUnimodular
        k : Nat
        f : Fin k → m
        g : Fin k → n
        hfg : And (Function.Injective f) (Function.Injective g)
        ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
      -/
    · exact hA k f g hfg.1 hfg.2
      /-
        🎉 no goals
      -/
      /-
        case neg
        m : Type u_1
        n : Type u_3
        R : Type u_5
        inst✝ : CommRing R
        A : Matrix m n R
        hA : A.IsTotallyUnimodular
        k : Nat
        f : Fin k → m
        g : Fin k → n
        hfg : Not (And (Function.Injective f) (Function.Injective g))
        ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
      -/
    · use 0
      /-
        case h
        m : Type u_1
        n : Type u_3
        R : Type u_5
        inst✝ : CommRing R
        A : Matrix m n R
        hA : A.IsTotallyUnimodular
        k : Nat
        f : Fin k → m
        g : Fin k → n
        hfg : Not (And (Function.Injective f) (Function.Injective g))
        ⊢ Eq (↑0) (A.submatrix f g).det
      -/
      rw [SignType.coe_zero, eq_comm]
      /-
        case h
        m : Type u_1
        n : Type u_3
        R : Type u_5
        inst✝ : CommRing R
        A : Matrix m n R
        hA : A.IsTotallyUnimodular
        k : Nat
        f : Fin k → m
        g : Fin k → n
        hfg : Not (And (Function.Injective f) (Function.Injective g))
        ⊢ Eq (A.submatrix f g).det 0
      -/
      simp_rw [not_and_or, Function.not_injective_iff] at hfg
      /-
        case h
        m : Type u_1
        n : Type u_3
        R : Type u_5
        inst✝ : CommRing R
        A : Matrix m n R
        hA : A.IsTotallyUnimodular
        k : Nat
        f : Fin k → m
        g : Fin k → n
        hfg : Or (Exists fun a => Exists fun b => And (Eq (f a) (f b)) (Ne a b)) (Exis …
        ⊢ Eq (A.submatrix f g).det 0
      -/
      obtain ⟨i, j, hfij, hij⟩ | ⟨i, j, hgij, hij⟩ := hfg
        /-
          case h.inl.intro.intro.intro
          m : Type u_1
          n : Type u_3
          R : Type u_5
          inst✝ : CommRing R
          A : Matrix m n R
          hA : A.IsTotallyUnimodular
          k : Nat
          f : Fin k → m
          g : Fin k → n
          i j : Fin k
          hfij : Eq (f i) (f j)
          hij : Ne i j
          ⊢ Eq (A.submatrix f g).det 0
        -/
      · rw [← det_transpose, transpose_submatrix]
        /-
          case h.inl.intro.intro.intro
          m : Type u_1
          n : Type u_3
          R : Type u_5
          inst✝ : CommRing R
          A : Matrix m n R
          hA : A.IsTotallyUnimodular
          k : Nat
          f : Fin k → m
          g : Fin k → n
          i j : Fin k
          hfij : Eq (f i) (f j)
          hij : Ne i j
          ⊢ Eq (A.transpose.submatrix g f).det 0
        -/
        apply det_zero_of_column_eq hij.symm
        /-
          case h.inl.intro.intro.intro
          m : Type u_1
          n : Type u_3
          R : Type u_5
          inst✝ : CommRing R
          A : Matrix m n R
          hA : A.IsTotallyUnimodular
          k : Nat
          f : Fin k → m
          g : Fin k → n
          i j : Fin k
          hfij : Eq (f i) (f j)
          hij : Ne i j
          ⊢ ∀ (k_1 : Fin k), Eq (A.transpose.submatrix g f k_1 j) (A.transpose.submatrix …
        -/
        simp [hfij]
        /-
          🎉 no goals
        -/
        /-
          case h.inr.intro.intro.intro
          m : Type u_1
          n : Type u_3
          R : Type u_5
          inst✝ : CommRing R
          A : Matrix m n R
          hA : A.IsTotallyUnimodular
          k : Nat
          f : Fin k → m
          g : Fin k → n
          i j : Fin k
          hgij : Eq (g i) (g j)
          hij : Ne i j
          ⊢ Eq (A.submatrix f g).det 0
        -/
      · apply det_zero_of_column_eq hij
        /-
          case h.inr.intro.intro.intro
          m : Type u_1
          n : Type u_3
          R : Type u_5
          inst✝ : CommRing R
          A : Matrix m n R
          hA : A.IsTotallyUnimodular
          k : Nat
          f : Fin k → m
          g : Fin k → n
          i j : Fin k
          hgij : Eq (g i) (g j)
          hij : Ne i j
          ⊢ ∀ (k_1 : Fin k), Eq (A.submatrix f g k_1 i) (A.submatrix f g k_1 j)
        -/
        simp [hgij]
        /-
          🎉 no goals
        -/
    /-
      case mpr
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
      ⊢ A.IsTotallyUnimodular
    -/
  · intro _ _ _ _ _
    /-
      case mpr
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
      k✝ : Nat
      f✝ : Fin k✝ → m
      g✝ : Fin k✝ → n
      a✝¹ : Function.Injective f✝
      a✝ : Function.Injective g✝
      ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f✝ g✝).det
    -/
    apply hA
    /-
      🎉 no goals
    -/


lemma isTotallyUnimodular_iff_fintype.{w} (A : Matrix m n R) : A.IsTotallyUnimodular ↔
    ∀ (ι : Type w) [Fintype ι] [DecidableEq ι], ∀ f : ι → m, ∀ g : ι → n,
      (A.submatrix f g).det ∈ Set.range SignType.cast := by
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    ⊢ Iff A.IsTotallyUnimodular (∀ (ι : Type w) [inst : Fintype ι] [inst_1 : Decid …
  -/
  rw [isTotallyUnimodular_iff]
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    ⊢ Iff (∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range  …
  -/
  constructor
    /-
      case mp
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      ⊢ (∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Sign …
    -/
  · intro hA ι _ _ f g
    /-
      case mp
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝² : CommRing R
      A : Matrix m n R
      hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
      ι : Type w
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : ι → m
      g : ι → n
      ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
    -/
    specialize hA (Fintype.card ι) (f ∘ (Fintype.equivFin ι).symm) (g ∘ (Fintype.equivFin ι).symm)
    /-
      case mp
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝² : CommRing R
      A : Matrix m n R
      ι : Type w
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : ι → m
      g : ι → n
      hA : Membership.mem (Set.range SignType.cast) (A.submatrix (Function.comp f ⇑( …
      ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
    -/
    rwa [←submatrix_submatrix, det_submatrix_equiv_self] at hA
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      ⊢ (∀ (ι : Type w) [inst : Fintype ι] [inst_1 : DecidableEq ι] (f : ι → m) (g : …
    -/
  · intro hA k f g
    /-
      case mpr
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      hA : ∀ (ι : Type w) [inst : Fintype ι] [inst_1 : DecidableEq ι] (f : ι → m) (g …
      k : Nat
      f : Fin k → m
      g : Fin k → n
      ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
    -/
    specialize hA (ULift (Fin k)) (f ∘ Equiv.ulift) (g ∘ Equiv.ulift)
    /-
      case mpr
      m : Type u_1
      n : Type u_3
      R : Type u_5
      inst✝ : CommRing R
      A : Matrix m n R
      k : Nat
      f : Fin k → m
      g : Fin k → n
      hA : Membership.mem (Set.range SignType.cast) (A.submatrix (Function.comp f ⇑E …
      ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix f g).det
    -/
    rwa [←submatrix_submatrix, det_submatrix_equiv_self] at hA
    /-
      🎉 no goals
    -/


lemma IsTotallyUnimodular.apply {A : Matrix m n R} (hA : A.IsTotallyUnimodular) (i : m) (j : n) :
    A i j ∈ Set.range SignType.cast := by
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    hA : A.IsTotallyUnimodular
    i : m
    j : n
    ⊢ Membership.mem (Set.range SignType.cast) (A i j)
  -/
  rw [isTotallyUnimodular_iff] at hA
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
    i : m
    j : n
    ⊢ Membership.mem (Set.range SignType.cast) (A i j)
  -/
  simpa using hA 1 (fun _ => i) (fun _ => j)
  /-
    🎉 no goals
  -/


lemma IsTotallyUnimodular.submatrix {A : Matrix m n R} (f : m' → m) (g : n' → n)
    (hA : A.IsTotallyUnimodular) :
    (A.submatrix f g).IsTotallyUnimodular := by
  /-
    m : Type u_1
    m' : Type u_2
    n : Type u_3
    n' : Type u_4
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    f : m' → m
    g : n' → n
    hA : A.IsTotallyUnimodular
    ⊢ (A.submatrix f g).IsTotallyUnimodular
  -/
  simp only [isTotallyUnimodular_iff, submatrix_submatrix] at hA ⊢
  /-
    m : Type u_1
    m' : Type u_2
    n : Type u_3
    n' : Type u_4
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    f : m' → m
    g : n' → n
    hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
    ⊢ ∀ (k : Nat) (f_1 : Fin k → m') (g_1 : Fin k → n'), Membership.mem (Set.range …
  -/
  intro _ _ _
  /-
    m : Type u_1
    m' : Type u_2
    n : Type u_3
    n' : Type u_4
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    f : m' → m
    g : n' → n
    hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
    k✝ : Nat
    f✝ : Fin k✝ → m'
    g✝ : Fin k✝ → n'
    ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix (Function.comp f f✝) ( …
  -/
  apply hA
  /-
    🎉 no goals
  -/


lemma IsTotallyUnimodular.transpose {A : Matrix m n R} (hA : A.IsTotallyUnimodular) :
    Aᵀ.IsTotallyUnimodular := by
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    hA : A.IsTotallyUnimodular
    ⊢ A.transpose.IsTotallyUnimodular
  -/
  simp only [isTotallyUnimodular_iff, ← transpose_submatrix, det_transpose] at hA ⊢
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
    ⊢ ∀ (k : Nat) (f : Fin k → n) (g : Fin k → m), Membership.mem (Set.range SignT …
  -/
  intro _ _ _
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    hA : ∀ (k : Nat) (f : Fin k → m) (g : Fin k → n), Membership.mem (Set.range Si …
    k✝ : Nat
    f✝ : Fin k✝ → n
    g✝ : Fin k✝ → m
    ⊢ Membership.mem (Set.range SignType.cast) (A.submatrix g✝ f✝).det
  -/
  apply hA
  /-
    🎉 no goals
  -/


lemma transpose_isTotallyUnimodular_iff (A : Matrix m n R) :
    Aᵀ.IsTotallyUnimodular ↔ A.IsTotallyUnimodular := by
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝ : CommRing R
    A : Matrix m n R
    ⊢ Iff A.transpose.IsTotallyUnimodular A.IsTotallyUnimodular
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> apply IsTotallyUnimodular.transpose
                  /-
                    🎉 no goals
                  -/


lemma IsTotallyUnimodular.reindex {A : Matrix m n R} (em : m ≃ m') (en : n ≃ n')
    (hA : A.IsTotallyUnimodular) :
    (A.reindex em en).IsTotallyUnimodular :=
  hA.submatrix _ _


lemma reindex_isTotallyUnimodular (A : Matrix m n R) (em : m ≃ m') (en : n ≃ n') :
    (A.reindex em en).IsTotallyUnimodular ↔ A.IsTotallyUnimodular :=
                /-
                  m : Type u_1
                  m' : Type u_2
                  n : Type u_3
                  n' : Type u_4
                  R : Type u_5
                  inst✝ : CommRing R
                  A : Matrix m n R
                  em : Equiv m m'
                  en : Equiv n n'
                  hA : ((Matrix.reindex em en) A).IsTotallyUnimodular
                  ⊢ A.IsTotallyUnimodular
                -/
  ⟨fun hA => by simpa [Equiv.symm_apply_eq] using hA.reindex em.symm en.symm,
                /-
                  🎉 no goals
                -/
   fun hA => hA.reindex _ _⟩


/-- If `A` is totally unimodular and each row of `B` is all zeros except for at most a single `1` or
a single `-1` then `fromRows A B` is totally unimodular. -/
lemma IsTotallyUnimodular.fromRows_unitlike [DecidableEq n] {A : Matrix m n R} {B : Matrix m' n R}
    (hA : A.IsTotallyUnimodular)
    (hB : Nonempty n → ∀ i : m', ∃ j : n, ∃ s : SignType, B i = Pi.single j s.cast) :
    (fromRows A B).IsTotallyUnimodular := by
  /-
    m : Type u_1
    m' : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝¹ : CommRing R
    inst✝ : DecidableEq n
    A : Matrix m n R
    B : Matrix m' n R
    hA : A.IsTotallyUnimodular
    hB : Nonempty n → ∀ (i : m'), Exists fun j => Exists fun s => Eq (B i) (Pi.sin …
    ⊢ (A.fromRows B).IsTotallyUnimodular
  -/
  intro k f g hf hg
  induction k with
  | zero => use 1; simp
  | succ k ih =>
    specialize hB ⟨g 0⟩
    -- Either `f` is `inr` somewhere or `inl` everywhere
    obtain ⟨i, j, hfi⟩ | ⟨f', rfl⟩ : (∃ i j, f i = .inr j) ∨ (∃ f', f = .inl ∘ f') := by
      simp_rw [← Sum.isRight_iff, or_iff_not_imp_left, not_exists, Bool.not_eq_true,
        Sum.isRight_eq_false, Sum.isLeft_iff]
      intro hfr
      choose f' hf' using hfr
      exact ⟨f', funext hf'⟩
    · have hAB := det_succ_row ((fromRows A B).submatrix f g) i
      simp only [submatrix_apply, hfi, fromRows_apply_inr] at hAB
      obtain ⟨j', s, hj'⟩ := hB j
      · simp only [hj', Function.update_apply] at hAB
        by_cases hj'' : ∃ x, g x = j'
        · obtain ⟨x, rfl⟩ := hj''
          rw [Fintype.sum_eq_single x fun y hxy => ?_, Pi.single_eq_same] at hAB
          · rw [hAB]
            change _ ∈ MonoidHom.mrange SignType.castHom.toMonoidHom
            refine mul_mem (mul_mem ?_ (Set.mem_range_self s)) ?_
            · apply pow_mem
              exact ⟨-1, by simp⟩
            · exact ih _ _
                (hf.comp Fin.succAbove_right_injective)
                (hg.comp Fin.succAbove_right_injective)
          · simp [Pi.single_eq_of_ne, hg.ne_iff.mpr hxy]
        · rw [not_exists] at hj''
          use 0
          simpa [hj''] using hAB.symm
    · rw [isTotallyUnimodular_iff] at hA
      apply hA


/-- If `A` is totally unimodular and each row of `B` is all zeros except for at most a single `1`,
then `fromRows A B` is totally unimodular. -/
lemma fromRows_isTotallyUnimodular_iff_rows [DecidableEq n] {A : Matrix m n R} {B : Matrix m' n R}
    (hB : Nonempty n → ∀ i : m', ∃ j : n, ∃ s : SignType, B i = Pi.single j s.cast) :
    (fromRows A B).IsTotallyUnimodular ↔ A.IsTotallyUnimodular :=
  ⟨.submatrix Sum.inl id, fun hA => hA.fromRows_unitlike hB⟩


lemma fromRows_one_isTotallyUnimodular_iff [DecidableEq n] (A : Matrix m n R) :
    (fromRows A (1 : Matrix n n R)).IsTotallyUnimodular ↔ A.IsTotallyUnimodular :=
  fromRows_isTotallyUnimodular_iff_rows <| fun h i ↦
                             /-
                               m : Type u_1
                               n : Type u_3
                               R : Type u_5
                               inst✝¹ : CommRing R
                               inst✝ : DecidableEq n
                               A : Matrix m n R
                               h : Nonempty n
                               i j : n
                               ⊢ Eq (1 i j) (Pi.single i (↑1) j)
                             -/
    ⟨i, 1, funext fun j ↦ by simp [one_apply, Pi.single_apply, eq_comm]⟩
                             /-
                               🎉 no goals
                             -/


lemma one_fromRows_isTotallyUnimodular_iff [DecidableEq n] (A : Matrix m n R) :
    (fromRows (1 : Matrix n n R) A).IsTotallyUnimodular ↔ A.IsTotallyUnimodular := by
  have hA :
    fromRows (1 : Matrix n n R) A =
      (fromRows A (1 : Matrix n n R)).reindex (Equiv.sumComm m n) (Equiv.refl n) := by
    aesop
  /-
    m : Type u_1
    n : Type u_3
    R : Type u_5
    inst✝¹ : CommRing R
    inst✝ : DecidableEq n
    A : Matrix m n R
    hA : Eq (Matrix.fromRows 1 A) ((Matrix.reindex (Equiv.sumComm m n) (Equiv.refl …
    ⊢ Iff (Matrix.fromRows 1 A).IsTotallyUnimodular A.IsTotallyUnimodular
  -/
  rw [hA, reindex_isTotallyUnimodular, fromRows_one_isTotallyUnimodular_iff]
  /-
    🎉 no goals
  -/


lemma fromCols_one_isTotallyUnimodular_iff [DecidableEq m] (A : Matrix m n R) :
    (fromCols A (1 : Matrix m m R)).IsTotallyUnimodular ↔ A.IsTotallyUnimodular := by
  rw [←transpose_isTotallyUnimodular_iff, transpose_fromCols, transpose_one,
    fromRows_one_isTotallyUnimodular_iff, transpose_isTotallyUnimodular_iff]


@[deprecated (since := "2024-12-11")]
alias fromColumns_one_isTotallyUnimodular_iff := fromCols_one_isTotallyUnimodular_iff


lemma one_fromCols_isTotallyUnimodular_iff [DecidableEq m] (A : Matrix m n R) :
    (fromCols (1 : Matrix m m R) A).IsTotallyUnimodular ↔ A.IsTotallyUnimodular := by
  rw [←transpose_isTotallyUnimodular_iff, transpose_fromCols, transpose_one,
    one_fromRows_isTotallyUnimodular_iff, transpose_isTotallyUnimodular_iff]


@[deprecated (since := "2024-12-11")]
alias one_fromColumns_isTotallyUnimodular_iff := one_fromCols_isTotallyUnimodular_iff


alias ⟨_, IsTotallyUnimodular.fromRows_one⟩ := fromRows_one_isTotallyUnimodular_iff

alias ⟨_, IsTotallyUnimodular.one_fromRows⟩ := one_fromRows_isTotallyUnimodular_iff

alias ⟨_, IsTotallyUnimodular.fromCols_one⟩ := fromCols_one_isTotallyUnimodular_iff

alias ⟨_, IsTotallyUnimodular.one_fromCols⟩ := one_fromCols_isTotallyUnimodular_iff


lemma fromRows_row0_isTotallyUnimodular_iff (A : Matrix m n R) :
    (fromRows A (row m' 0)).IsTotallyUnimodular ↔ A.IsTotallyUnimodular := by
  classical
  refine fromRows_isTotallyUnimodular_iff_rows <| fun _ _ => ?_
  inhabit n
  refine ⟨default, 0, ?_⟩
  ext x
  simp [Pi.single_apply]


lemma fromCols_col0_isTotallyUnimodular_iff (A : Matrix m n R) :
    (fromCols A (col n' 0)).IsTotallyUnimodular ↔ A.IsTotallyUnimodular := by
  rw [← transpose_isTotallyUnimodular_iff, transpose_fromCols, transpose_col,
    fromRows_row0_isTotallyUnimodular_iff, transpose_isTotallyUnimodular_iff]


@[deprecated (since := "2024-12-11")]
alias fromColumns_col0_isTotallyUnimodular_iff := fromCols_col0_isTotallyUnimodular_iff


