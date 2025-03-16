/-- The next common fixed point, at least `a`, for a family of normal functions.

This is defined for any family of functions, as the supremum of all values reachable by applying
finitely many functions in the family to `a`.

`Ordinal.nfpFamily_fp` shows this is a fixed point, `Ordinal.le_nfpFamily` shows it's at
least `a`, and `Ordinal.nfpFamily_le_fp` shows this is the least ordinal with these properties. -/
def nfpFamily (f : ι → Ordinal.{u} → Ordinal.{u}) (a : Ordinal.{u}) : Ordinal :=
  ⨆ i, List.foldr f a i


@[deprecated "No deprecation message was provided." (since := "2024-10-14")]
theorem nfpFamily_eq_sup (f : ι → Ordinal.{u} → Ordinal.{u}) (a : Ordinal.{u}) :
    nfpFamily f a = ⨆ i, List.foldr f a i :=
  rfl


theorem foldr_le_nfpFamily [Small.{u} ι] (f : ι → Ordinal.{u} → Ordinal.{u}) (a l) :
    List.foldr f a l ≤ nfpFamily f a :=
  Ordinal.le_iSup _ _


theorem le_nfpFamily [Small.{u} ι] (f : ι → Ordinal.{u} → Ordinal.{u}) (a) : a ≤ nfpFamily f a :=
  foldr_le_nfpFamily f a []


theorem lt_nfpFamily [Small.{u} ι] {a b} : a < nfpFamily f b ↔ ∃ l, a < List.foldr f b l :=
  Ordinal.lt_iSup_iff


theorem nfpFamily_le_iff [Small.{u} ι] {a b} : nfpFamily f a ≤ b ↔ ∀ l, List.foldr f a l ≤ b :=
  Ordinal.iSup_le_iff


theorem nfpFamily_le {a b} : (∀ l, List.foldr f a l ≤ b) → nfpFamily f a ≤ b :=
  Ordinal.iSup_le


theorem nfpFamily_monotone [Small.{u} ι] (hf : ∀ i, Monotone (f i)) : Monotone (nfpFamily f) :=
  fun _ _ h ↦ nfpFamily_le <| fun l ↦ (List.foldr_monotone hf l h).trans (foldr_le_nfpFamily _ _ l)


theorem apply_lt_nfpFamily [Small.{u} ι] (H : ∀ i, IsNormal (f i)) {a b}
    (hb : b < nfpFamily f a) (i) : f i b < nfpFamily f a :=
  let ⟨l, hl⟩ := lt_nfpFamily.1 hb
  lt_nfpFamily.2 ⟨i::l, (H i).strictMono hl⟩


theorem apply_lt_nfpFamily_iff [Nonempty ι] [Small.{u} ι] (H : ∀ i, IsNormal (f i)) {a b} :
    (∀ i, f i b < nfpFamily f a) ↔ b < nfpFamily f a := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝¹ : Nonempty ι
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    ⊢ Iff (∀ (i : ι), LT.lt (f i b) (Ordinal.nfpFamily f a)) (LT.lt b (Ordinal.nfp …
  -/
  refine ⟨fun h ↦ ?_, apply_lt_nfpFamily H⟩
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝¹ : Nonempty ι
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    h : ∀ (i : ι), LT.lt (f i b) (Ordinal.nfpFamily f a)
    ⊢ LT.lt b (Ordinal.nfpFamily f a)
  -/
  let ⟨l, hl⟩ := lt_nfpFamily.1 (h (Classical.arbitrary ι))
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝¹ : Nonempty ι
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    h : ∀ (i : ι), LT.lt (f i b) (Ordinal.nfpFamily f a)
    l : List ι
    hl : LT.lt (f (Classical.arbitrary ι) b) (List.foldr f a l)
    ⊢ LT.lt b (Ordinal.nfpFamily f a)
  -/
  exact lt_nfpFamily.2 <| ⟨l, (H _).le_apply.trans_lt hl⟩
  /-
    🎉 no goals
  -/


theorem nfpFamily_le_apply [Nonempty ι] [Small.{u} ι] (H : ∀ i, IsNormal (f i)) {a b} :
    (∃ i, nfpFamily f a ≤ f i b) ↔ nfpFamily f a ≤ b := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝¹ : Nonempty ι
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    ⊢ Iff (Exists fun i => LE.le (Ordinal.nfpFamily f a) (f i b)) (LE.le (Ordinal. …
  -/
  rw [← not_iff_not]
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝¹ : Nonempty ι
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    ⊢ Iff (Not (Exists fun i => LE.le (Ordinal.nfpFamily f a) (f i b))) (Not (LE.l …
  -/
  push_neg
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝¹ : Nonempty ι
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    ⊢ Iff (∀ (i : ι), LT.lt (f i b) (Ordinal.nfpFamily f a)) (LT.lt b (Ordinal.nfp …
  -/
  exact apply_lt_nfpFamily_iff H
  /-
    🎉 no goals
  -/


theorem nfpFamily_le_fp (H : ∀ i, Monotone (f i)) {a b} (ab : a ≤ b) (h : ∀ i, f i b ≤ b) :
    nfpFamily f a ≤ b := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    H : ∀ (i : ι), Monotone (f i)
    a b : Ordinal.{u}
    ab : LE.le a b
    h : ∀ (i : ι), LE.le (f i b) b
    ⊢ LE.le (Ordinal.nfpFamily f a) b
  -/
  apply Ordinal.iSup_le
  /-
    case a
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    H : ∀ (i : ι), Monotone (f i)
    a b : Ordinal.{u}
    ab : LE.le a b
    h : ∀ (i : ι), LE.le (f i b) b
    ⊢ ∀ (i : List ι), LE.le (List.foldr f a i) b
  -/
  intro l
  /-
    case a
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    H : ∀ (i : ι), Monotone (f i)
    a b : Ordinal.{u}
    ab : LE.le a b
    h : ∀ (i : ι), LE.le (f i b) b
    l : List ι
    ⊢ LE.le (List.foldr f a l) b
  -/
  induction' l with i l IH generalizing a
    /-
      case a.nil
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      H : ∀ (i : ι), Monotone (f i)
      b : Ordinal.{u}
      h : ∀ (i : ι), LE.le (f i b) b
      a : Ordinal.{u}
      ab : LE.le a b
      ⊢ LE.le (List.foldr f a List.nil) b
    -/
  · exact ab
    /-
      🎉 no goals
    -/
    /-
      case a.cons
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      H : ∀ (i : ι), Monotone (f i)
      b : Ordinal.{u}
      h : ∀ (i : ι), LE.le (f i b) b
      i : ι
      l : List ι
      IH : ∀ {a : Ordinal.{u}}, LE.le a b → LE.le (List.foldr f a l) b
      a : Ordinal.{u}
      ab : LE.le a b
      ⊢ LE.le (List.foldr f a (List.cons i l)) b
    -/
  · exact (H i (IH ab)).trans (h i)
    /-
      🎉 no goals
    -/


theorem nfpFamily_fp [Small.{u} ι] {i} (H : IsNormal (f i)) (a) :
    f i (nfpFamily f a) = nfpFamily f a := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    i : ι
    H : Ordinal.IsNormal (f i)
    a : Ordinal.{u}
    ⊢ Eq (f i (Ordinal.nfpFamily f a)) (Ordinal.nfpFamily f a)
  -/
  rw [nfpFamily, H.map_iSup]
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    i : ι
    H : Ordinal.IsNormal (f i)
    a : Ordinal.{u}
    ⊢ Eq (iSup fun i_1 => f i (List.foldr f a i_1)) (iSup fun i => List.foldr f a i)
  -/
  apply le_antisymm <;> refine Ordinal.iSup_le fun l => ?_
    /-
      case a
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      a : Ordinal.{u}
      l : List ι
      ⊢ LE.le (f i (List.foldr f a l)) (iSup fun i => List.foldr f a i)
    -/
  · exact Ordinal.le_iSup _ (i::l)
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      a : Ordinal.{u}
      l : List ι
      ⊢ LE.le (List.foldr f a l) (iSup fun i_1 => f i (List.foldr f a i_1))
    -/
  · exact H.le_apply.trans (Ordinal.le_iSup _ _)
    /-
      🎉 no goals
    -/


theorem apply_le_nfpFamily [Small.{u} ι] [hι : Nonempty ι] (H : ∀ i, IsNormal (f i)) {a b} :
    (∀ i, f i b ≤ nfpFamily f a) ↔ b ≤ nfpFamily f a := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    hι : Nonempty ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a b : Ordinal.{u}
    ⊢ Iff (∀ (i : ι), LE.le (f i b) (Ordinal.nfpFamily f a)) (LE.le b (Ordinal.nfp …
  -/
  refine ⟨fun h => ?_, fun h i => ?_⟩
    /-
      case refine_1
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      hι : Nonempty ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a b : Ordinal.{u}
      h : ∀ (i : ι), LE.le (f i b) (Ordinal.nfpFamily f a)
      ⊢ LE.le b (Ordinal.nfpFamily f a)
    -/
  · obtain ⟨i⟩ := hι
    /-
      case refine_1.intro
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a b : Ordinal.{u}
      h : ∀ (i : ι), LE.le (f i b) (Ordinal.nfpFamily f a)
      i : ι
      ⊢ LE.le b (Ordinal.nfpFamily f a)
    -/
    exact (H i).le_apply.trans (h i)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      hι : Nonempty ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a b : Ordinal.{u}
      h : LE.le b (Ordinal.nfpFamily f a)
      i : ι
      ⊢ LE.le (f i b) (Ordinal.nfpFamily f a)
    -/
  · rw [← nfpFamily_fp (H i)]
    /-
      case refine_2
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      hι : Nonempty ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a b : Ordinal.{u}
      h : LE.le b (Ordinal.nfpFamily f a)
      i : ι
      ⊢ LE.le (f i b) (f i (Ordinal.nfpFamily f a))
    -/
    exact (H i).monotone h
    /-
      🎉 no goals
    -/


theorem nfpFamily_eq_self [Small.{u} ι] {a} (h : ∀ i, f i a = a) : nfpFamily f a = a := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    a : Ordinal.{u}
    h : ∀ (i : ι), Eq (f i a) a
    ⊢ Eq (Ordinal.nfpFamily f a) a
  -/
  apply (Ordinal.iSup_le ?_).antisymm (le_nfpFamily f a)
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    a : Ordinal.{u}
    h : ∀ (i : ι), Eq (f i a) a
    ⊢ ∀ (i : List ι), LE.le (List.foldr f a i) a
  -/
  intro l
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    a : Ordinal.{u}
    h : ∀ (i : ι), Eq (f i a) a
    l : List ι
    ⊢ LE.le (List.foldr f a l) a
  -/
  rw [List.foldr_fixed' h l]
  /-
    🎉 no goals
  -/

-- Todo: This is actually a special case of the fact the intersection of club sets is a club set.

/-- A generalization of the fixed point lemma for normal functions: any family of normal functions
    has an unbounded set of common fixed points. -/
theorem not_bddAbove_fp_family [Small.{u} ι] (H : ∀ i, IsNormal (f i)) :
    ¬ BddAbove (⋂ i, Function.fixedPoints (f i)) := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    ⊢ Not (BddAbove (Set.iInter fun i => Function.fixedPoints (f i)))
  -/
  rw [not_bddAbove_iff]
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    ⊢ ∀ (x : Ordinal.{u}), Exists fun y => And (Membership.mem (Set.iInter fun i = …
  -/
  refine fun a ↦ ⟨nfpFamily f (succ a), ?_, (lt_succ a).trans_le (le_nfpFamily f _)⟩
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a : Ordinal.{u}
    ⊢ Membership.mem (Set.iInter fun i => Function.fixedPoints (f i)) (Ordinal.nfp …
  -/
  rintro _ ⟨i, rfl⟩
  /-
    case intro
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a : Ordinal.{u}
    i : ι
    ⊢ Membership.mem ((fun i => Function.fixedPoints (f i)) i) (Ordinal.nfpFamily  …
  -/
  exact nfpFamily_fp (H i) _
  /-
    🎉 no goals
  -/


/-- The derivative of a family of normal functions is the sequence of their common fixed points.

This is defined for all functions such that `Ordinal.derivFamily_zero`,
`Ordinal.derivFamily_succ`, and `Ordinal.derivFamily_limit` are satisfied. -/
def derivFamily (f : ι → Ordinal.{u} → Ordinal.{u}) (o : Ordinal.{u}) : Ordinal.{u} :=
  limitRecOn o (nfpFamily f 0) (fun _ IH => nfpFamily f (succ IH))
    fun a _ g => ⨆ b : Set.Iio a, g _ b.2


@[simp]
theorem derivFamily_zero (f : ι → Ordinal → Ordinal) :
    derivFamily f 0 = nfpFamily f 0 :=
  limitRecOn_zero ..


@[simp]
theorem derivFamily_succ (f : ι → Ordinal → Ordinal) (o) :
    derivFamily f (succ o) = nfpFamily f (succ (derivFamily f o)) :=
  limitRecOn_succ ..


theorem derivFamily_limit (f : ι → Ordinal → Ordinal) {o} :
    IsLimit o → derivFamily f o = ⨆ b : Set.Iio o, derivFamily f b :=
  limitRecOn_limit _ _ _ _


theorem isNormal_derivFamily [Small.{u} ι] (f : ι → Ordinal.{u} → Ordinal.{u}) :
    IsNormal (derivFamily f) := by
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Ordinal.{u} → Ordinal.{u}
    ⊢ Ordinal.IsNormal (Ordinal.derivFamily f)
  -/
  refine ⟨fun o ↦ ?_, fun o h a ↦ ?_⟩
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Small.{u, u_1} ι
      f : ι → Ordinal.{u} → Ordinal.{u}
      o : Ordinal.{u}
      ⊢ LT.lt (Ordinal.derivFamily f o) (Ordinal.derivFamily f (Order.succ o))
    -/
  · rw [derivFamily_succ, ← succ_le_iff]
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Small.{u, u_1} ι
      f : ι → Ordinal.{u} → Ordinal.{u}
      o : Ordinal.{u}
      ⊢ LE.le (Order.succ (Ordinal.derivFamily f o)) (Ordinal.nfpFamily f (Order.suc …
    -/
    exact le_nfpFamily _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Small.{u, u_1} ι
      f : ι → Ordinal.{u} → Ordinal.{u}
      o : Ordinal.{u}
      h : o.IsLimit
      a : Ordinal.{u}
      ⊢ Iff (LE.le (Ordinal.derivFamily f o) a) (∀ (b : Ordinal.{u}), LT.lt b o → LE …
    -/
  · simp_rw [derivFamily_limit _ h, Ordinal.iSup_le_iff, Subtype.forall, Set.mem_Iio]
    /-
      🎉 no goals
    -/


@[deprecated isNormal_derivFamily (since := "2024-10-11")]
alias derivFamily_isNormal := isNormal_derivFamily


theorem derivFamily_fp [Small.{u} ι] {i} (H : IsNormal (f i)) (o : Ordinal) :
    f i (derivFamily f o) = derivFamily f o := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    i : ι
    H : Ordinal.IsNormal (f i)
    o : Ordinal.{u}
    ⊢ Eq (f i (Ordinal.derivFamily f o)) (Ordinal.derivFamily f o)
  -/
  induction' o using limitRecOn with o _ o l IH
    /-
      case H₁
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      ⊢ Eq (f i (Ordinal.derivFamily f 0)) (Ordinal.derivFamily f 0)
    -/
  · rw [derivFamily_zero]
    /-
      case H₁
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      ⊢ Eq (f i (Ordinal.nfpFamily f 0)) (Ordinal.nfpFamily f 0)
    -/
    exact nfpFamily_fp H 0
    /-
      🎉 no goals
    -/
    /-
      case H₂
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      a✝ : Eq (f i (Ordinal.derivFamily f o)) (Ordinal.derivFamily f o)
      ⊢ Eq (f i (Ordinal.derivFamily f (Order.succ o))) (Ordinal.derivFamily f (Orde …
    -/
  · rw [derivFamily_succ]
    /-
      case H₂
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      a✝ : Eq (f i (Ordinal.derivFamily f o)) (Ordinal.derivFamily f o)
      ⊢ Eq (f i (Ordinal.nfpFamily f (Order.succ (Ordinal.derivFamily f o)))) (Ordin …
    -/
    exact nfpFamily_fp H _
    /-
      🎉 no goals
    -/
    /-
      case H₃
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      l : o.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → Eq (f i (Ordinal.derivFamily f o')) (O …
      ⊢ Eq (f i (Ordinal.derivFamily f o)) (Ordinal.derivFamily f o)
    -/
  · have : Nonempty (Set.Iio o) := ⟨0, l.pos⟩
    /-
      case H₃
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      l : o.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → Eq (f i (Ordinal.derivFamily f o')) (O …
      this : Nonempty ↑(Set.Iio o)
      ⊢ Eq (f i (Ordinal.derivFamily f o)) (Ordinal.derivFamily f o)
    -/
    rw [derivFamily_limit _ l, H.map_iSup]
    /-
      case H₃
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      l : o.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → Eq (f i (Ordinal.derivFamily f o')) (O …
      this : Nonempty ↑(Set.Iio o)
      ⊢ Eq (iSup fun i_1 => f i (Ordinal.derivFamily f ↑i_1)) (iSup fun b => Ordinal …
    -/
    refine eq_of_forall_ge_iff fun c => ?_
    /-
      case H₃
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      l : o.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → Eq (f i (Ordinal.derivFamily f o')) (O …
      this : Nonempty ↑(Set.Iio o)
      c : Ordinal.{u}
      ⊢ Iff (LE.le (iSup fun i_1 => f i (Ordinal.derivFamily f ↑i_1)) c) (LE.le (iSu …
    -/
    rw [Ordinal.iSup_le_iff, Ordinal.iSup_le_iff]
    /-
      case H₃
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      l : o.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → Eq (f i (Ordinal.derivFamily f o')) (O …
      this : Nonempty ↑(Set.Iio o)
      c : Ordinal.{u}
      ⊢ Iff (∀ (i_1 : ↑(Set.Iio o)), LE.le (f i (Ordinal.derivFamily f ↑i_1)) c) (∀  …
    -/
    refine forall_congr' fun a ↦ ?_
    /-
      case H₃
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      i : ι
      H : Ordinal.IsNormal (f i)
      o : Ordinal.{u}
      l : o.IsLimit
      IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → Eq (f i (Ordinal.derivFamily f o')) (O …
      this : Nonempty ↑(Set.Iio o)
      c : Ordinal.{u}
      a : ↑(Set.Iio o)
      ⊢ Iff (LE.le (f i (Ordinal.derivFamily f ↑a)) c) (LE.le (Ordinal.derivFamily f …
    -/
    rw [IH _ a.2]
    /-
      🎉 no goals
    -/


theorem le_iff_derivFamily [Small.{u} ι] (H : ∀ i, IsNormal (f i)) {a} :
    (∀ i, f i a ≤ a) ↔ ∃ o, derivFamily f o = a :=
  ⟨fun ha => by
    suffices ∀ (o), a ≤ derivFamily f o → ∃ o, derivFamily f o = a from
      this a (isNormal_derivFamily _).le_apply
    /-
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a : Ordinal.{u}
      ha : ∀ (i : ι), LE.le (f i a) a
      ⊢ ∀ (o : Ordinal.{u}), LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq  …
    -/
    intro o
    /-
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a : Ordinal.{u}
      ha : ∀ (i : ι), LE.le (f i a) a
      o : Ordinal.{u}
      ⊢ LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFamily  …
    -/
    induction' o using limitRecOn with o IH o l IH
      /-
        case H₁
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        ⊢ LE.le a (Ordinal.derivFamily f 0) → Exists fun o => Eq (Ordinal.derivFamily  …
      -/
    · intro h₁
      /-
        case H₁
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        h₁ : LE.le a (Ordinal.derivFamily f 0)
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      refine ⟨0, le_antisymm ?_ h₁⟩
      /-
        case H₁
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        h₁ : LE.le a (Ordinal.derivFamily f 0)
        ⊢ LE.le (Ordinal.derivFamily f 0) a
      -/
      rw [derivFamily_zero]
      /-
        case H₁
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        h₁ : LE.le a (Ordinal.derivFamily f 0)
        ⊢ LE.le (Ordinal.nfpFamily f 0) a
      -/
      exact nfpFamily_le_fp (fun i => (H i).monotone) (Ordinal.zero_le _) ha
      /-
        🎉 no goals
      -/
      /-
        case H₂
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        IH : LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFami …
        ⊢ LE.le a (Ordinal.derivFamily f (Order.succ o)) → Exists fun o => Eq (Ordinal …
      -/
    · intro h₁
      /-
        case H₂
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        IH : LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFami …
        h₁ : LE.le a (Ordinal.derivFamily f (Order.succ o))
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      rcases le_or_lt a (derivFamily f o) with h | h
        /-
          case H₂.inl
          ι : Type u_1
          f : ι → Ordinal.{u} → Ordinal.{u}
          inst✝ : Small.{u, u_1} ι
          H : ∀ (i : ι), Ordinal.IsNormal (f i)
          a : Ordinal.{u}
          ha : ∀ (i : ι), LE.le (f i a) a
          o : Ordinal.{u}
          IH : LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFami …
          h₁ : LE.le a (Ordinal.derivFamily f (Order.succ o))
          h : LE.le a (Ordinal.derivFamily f o)
          ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
        -/
      · exact IH h
        /-
          🎉 no goals
        -/
      /-
        case H₂.inr
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        IH : LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFami …
        h₁ : LE.le a (Ordinal.derivFamily f (Order.succ o))
        h : LT.lt (Ordinal.derivFamily f o) a
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      refine ⟨succ o, le_antisymm ?_ h₁⟩
      /-
        case H₂.inr
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        IH : LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFami …
        h₁ : LE.le a (Ordinal.derivFamily f (Order.succ o))
        h : LT.lt (Ordinal.derivFamily f o) a
        ⊢ LE.le (Ordinal.derivFamily f (Order.succ o)) a
      -/
      rw [derivFamily_succ]
      /-
        case H₂.inr
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        IH : LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFami …
        h₁ : LE.le a (Ordinal.derivFamily f (Order.succ o))
        h : LT.lt (Ordinal.derivFamily f o) a
        ⊢ LE.le (Ordinal.nfpFamily f (Order.succ (Ordinal.derivFamily f o))) a
      -/
      exact nfpFamily_le_fp (fun i => (H i).monotone) (succ_le_of_lt h) ha
      /-
        🎉 no goals
      -/
      /-
        case H₃
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        l : o.IsLimit
        IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le a (Ordinal.derivFamily f o') → E …
        ⊢ LE.le a (Ordinal.derivFamily f o) → Exists fun o => Eq (Ordinal.derivFamily  …
      -/
    · intro h₁
      /-
        case H₃
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        l : o.IsLimit
        IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le a (Ordinal.derivFamily f o') → E …
        h₁ : LE.le a (Ordinal.derivFamily f o)
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      cases' eq_or_lt_of_le h₁ with h h
        /-
          case H₃.inl
          ι : Type u_1
          f : ι → Ordinal.{u} → Ordinal.{u}
          inst✝ : Small.{u, u_1} ι
          H : ∀ (i : ι), Ordinal.IsNormal (f i)
          a : Ordinal.{u}
          ha : ∀ (i : ι), LE.le (f i a) a
          o : Ordinal.{u}
          l : o.IsLimit
          IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le a (Ordinal.derivFamily f o') → E …
          h₁ : LE.le a (Ordinal.derivFamily f o)
          h : Eq a (Ordinal.derivFamily f o)
          ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
        -/
      · exact ⟨_, h.symm⟩
        /-
          🎉 no goals
        -/
      /-
        case H₃.inr
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        l : o.IsLimit
        IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le a (Ordinal.derivFamily f o') → E …
        h₁ : LE.le a (Ordinal.derivFamily f o)
        h : LT.lt a (Ordinal.derivFamily f o)
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      rw [derivFamily_limit _ l, ← not_le, Ordinal.iSup_le_iff, not_forall] at h
      /-
        case H₃.inr
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        l : o.IsLimit
        IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le a (Ordinal.derivFamily f o') → E …
        h₁ : LE.le a (Ordinal.derivFamily f o)
        h : Exists fun x => Not (LE.le (Ordinal.derivFamily f ↑x) a)
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      obtain ⟨o', h⟩ := h
      /-
        case H₃.inr.intro
        ι : Type u_1
        f : ι → Ordinal.{u} → Ordinal.{u}
        inst✝ : Small.{u, u_1} ι
        H : ∀ (i : ι), Ordinal.IsNormal (f i)
        a : Ordinal.{u}
        ha : ∀ (i : ι), LE.le (f i a) a
        o : Ordinal.{u}
        l : o.IsLimit
        IH : ∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le a (Ordinal.derivFamily f o') → E …
        h₁ : LE.le a (Ordinal.derivFamily f o)
        o' : ↑(Set.Iio o)
        h : Not (LE.le (Ordinal.derivFamily f ↑o') a)
        ⊢ Exists fun o => Eq (Ordinal.derivFamily f o) a
      -/
      exact IH o' o'.2 (le_of_not_le h),
      /-
        🎉 no goals
      -/
    fun ⟨_, e⟩ i => e ▸ (derivFamily_fp (H i) _).le⟩


theorem fp_iff_derivFamily [Small.{u} ι] (H : ∀ i, IsNormal (f i)) {a} :
    (∀ i, f i a = a) ↔ ∃ o, derivFamily f o = a :=
  Iff.trans ⟨fun h i => le_of_eq (h i), fun h i => (H i).le_iff_eq.1 (h i)⟩ (le_iff_derivFamily H)


/-- For a family of normal functions, `Ordinal.derivFamily` enumerates the common fixed points. -/
theorem derivFamily_eq_enumOrd [Small.{u} ι] (H : ∀ i, IsNormal (f i)) :
    derivFamily f = enumOrd (⋂ i, Function.fixedPoints (f i)) := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    ⊢ Eq (Ordinal.derivFamily f) (Ordinal.enumOrd (Set.iInter fun i => Function.fi …
  -/
  rw [eq_comm, eq_enumOrd _ (not_bddAbove_fp_family H)]
  /-
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    ⊢ And (StrictMono (Ordinal.derivFamily f)) (Eq (Set.range (Ordinal.derivFamily …
  -/
  use (isNormal_derivFamily f).strictMono
  /-
    case right
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    ⊢ Eq (Set.range (Ordinal.derivFamily f)) (Set.iInter fun i => Function.fixedPo …
  -/
  rw [Set.range_eq_iff]
  /-
    case right
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    ⊢ And (∀ (a : Ordinal.{u}), Membership.mem (Set.iInter fun i => Function.fixed …
  -/
  refine ⟨?_, fun a ha => ?_⟩
    /-
      case right.refine_1
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      ⊢ ∀ (a : Ordinal.{u}), Membership.mem (Set.iInter fun i => Function.fixedPoint …
    -/
  · rintro a S ⟨i, hi⟩
    /-
      case right.refine_1.intro
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a : Ordinal.{u}
      S : Set Ordinal.{u}
      i : ι
      hi : Eq ((fun i => Function.fixedPoints (f i)) i) S
      ⊢ Membership.mem S (Ordinal.derivFamily f a)
    -/
    rw [← hi]
    /-
      case right.refine_1.intro
      ι : Type u_1
      f : ι → Ordinal.{u} → Ordinal.{u}
      inst✝ : Small.{u, u_1} ι
      H : ∀ (i : ι), Ordinal.IsNormal (f i)
      a : Ordinal.{u}
      S : Set Ordinal.{u}
      i : ι
      hi : Eq ((fun i => Function.fixedPoints (f i)) i) S
      ⊢ Membership.mem ((fun i => Function.fixedPoints (f i)) i) (Ordinal.derivFamil …
    -/
    exact derivFamily_fp (H i) a
    /-
      🎉 no goals
    -/
  /-
    case right.refine_2
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a : Ordinal.{u}
    ha : Membership.mem (Set.iInter fun i => Function.fixedPoints (f i)) a
    ⊢ Exists fun a_1 => Eq (Ordinal.derivFamily f a_1) a
  -/
  rw [Set.mem_iInter] at ha
  /-
    case right.refine_2
    ι : Type u_1
    f : ι → Ordinal.{u} → Ordinal.{u}
    inst✝ : Small.{u, u_1} ι
    H : ∀ (i : ι), Ordinal.IsNormal (f i)
    a : Ordinal.{u}
    ha : ∀ (i : ι), Membership.mem (Function.fixedPoints (f i)) a
    ⊢ Exists fun a_1 => Eq (Ordinal.derivFamily f a_1) a
  -/
  rwa [← fp_iff_derivFamily H]
  /-
    🎉 no goals
  -/


/-- The next common fixed point, at least `a`, for a family of normal functions indexed by ordinals.

This is defined as `Ordinal.nfpFamily` of the type-indexed family associated to `f`. -/
@[deprecated nfpFamily (since := "2024-10-14")]
def nfpBFamily (o : Ordinal.{u}) (f : ∀ b < o, Ordinal.{max u v} → Ordinal.{max u v}) :
    Ordinal.{max u v} → Ordinal.{max u v} :=
  nfpFamily (familyOfBFamily o f)


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_eq_nfpFamily {o : Ordinal} (f : ∀ b < o, Ordinal → Ordinal) :
    nfpBFamily.{u, v} o f = nfpFamily (familyOfBFamily o f) :=
  rfl


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem foldr_le_nfpBFamily {o : Ordinal}
    (f : ∀ b < o, Ordinal → Ordinal) (a l) :
    List.foldr (familyOfBFamily o f) a l ≤ nfpBFamily.{u, v} o f a :=
  Ordinal.le_iSup _ _


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem le_nfpBFamily {o : Ordinal} (f : ∀ b < o, Ordinal → Ordinal) (a) :
    a ≤ nfpBFamily.{u, v} o f a :=
  Ordinal.le_iSup (fun _ ↦ List.foldr _ a _) []


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem lt_nfpBFamily {a b} :
    a < nfpBFamily.{u, v} o f b ↔ ∃ l, a < List.foldr (familyOfBFamily o f) b l :=
  Ordinal.lt_iSup_iff


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_le_iff {o : Ordinal} {f : ∀ b < o, Ordinal → Ordinal} {a b} :
    nfpBFamily.{u, v} o f a ≤ b ↔ ∀ l, List.foldr (familyOfBFamily o f) a l ≤ b :=
  Ordinal.iSup_le_iff


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_le {o : Ordinal} {f : ∀ b < o, Ordinal → Ordinal} {a b} :
    (∀ l, List.foldr (familyOfBFamily o f) a l ≤ b) → nfpBFamily.{u, v} o f a ≤ b :=
  Ordinal.iSup_le


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_monotone (hf : ∀ i hi, Monotone (f i hi)) : Monotone (nfpBFamily.{u, v} o f) :=
  nfpFamily_monotone fun _ => hf _ _


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem apply_lt_nfpBFamily (H : ∀ i hi, IsNormal (f i hi)) {a b} (hb : b < nfpBFamily.{u, v} o f a)
    (i hi) : f i hi b < nfpBFamily.{u, v} o f a := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a b : Ordinal.{max u v}
    hb : LT.lt b (o.nfpBFamily f a)
    i : Ordinal.{u}
    hi : LT.lt i o
    ⊢ LT.lt (f i hi b) (o.nfpBFamily f a)
  -/
  rw [← familyOfBFamily_enum o f]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a b : Ordinal.{max u v}
    hb : LT.lt b (o.nfpBFamily f a)
    i : Ordinal.{u}
    hi : LT.lt i o
    ⊢ LT.lt (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨i, ⋯⟩)  …
  -/
  apply apply_lt_nfpFamily (fun _ => H _ _) hb
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem apply_lt_nfpBFamily_iff (ho : o ≠ 0) (H : ∀ i hi, IsNormal (f i hi)) {a b} :
    (∀ i hi, f i hi b < nfpBFamily.{u, v} o f a) ↔ b < nfpBFamily.{u, v} o f a :=
  ⟨fun h => by
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi b) (o.nfpBFamily f a)
      ⊢ LT.lt b (o.nfpBFamily f a)
    -/
    haveI := toType_nonempty_iff_ne_zero.2 ho
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi b) (o.nfpBFamily f a)
      this : Nonempty o.toType
      ⊢ LT.lt b (o.nfpBFamily f a)
    -/
    refine (apply_lt_nfpFamily_iff ?_).1 fun _ => h _ _
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi b) (o.nfpBFamily f a)
      this : Nonempty o.toType
      ⊢ ∀ (i : o.toType), Ordinal.IsNormal (o.familyOfBFamily f i)
    -/
    exact fun _ => H _ _, apply_lt_nfpBFamily H⟩
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_le_apply (ho : o ≠ 0) (H : ∀ i hi, IsNormal (f i hi)) {a b} :
    (∃ i hi, nfpBFamily.{u, v} o f a ≤ f i hi b) ↔ nfpBFamily.{u, v} o f a ≤ b := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    ho : Ne o 0
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a b : Ordinal.{max u v}
    ⊢ Iff (Exists fun i => Exists fun hi => LE.le (o.nfpBFamily f a) (f i hi b)) ( …
  -/
  rw [← not_iff_not]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    ho : Ne o 0
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a b : Ordinal.{max u v}
    ⊢ Iff (Not (Exists fun i => Exists fun hi => LE.le (o.nfpBFamily f a) (f i hi  …
  -/
  push_neg
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    ho : Ne o 0
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a b : Ordinal.{max u v}
    ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi b) (o.nfpBFamily f  …
  -/
  exact apply_lt_nfpBFamily_iff.{u, v} ho H
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_le_fp (H : ∀ i hi, Monotone (f i hi)) {a b} (ab : a ≤ b)
    (h : ∀ i hi, f i hi b ≤ b) : nfpBFamily.{u, v} o f a ≤ b :=
  nfpFamily_le_fp (fun _ => H _ _) ab fun _ => h _ _


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_fp {i hi} (H : IsNormal (f i hi)) (a) :
    f i hi (nfpBFamily.{u, v} o f a) = nfpBFamily.{u, v} o f a := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Eq (f i hi (o.nfpBFamily f a)) (o.nfpBFamily f a)
  -/
  rw [← familyOfBFamily_enum o f]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Eq (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨i, ⋯⟩) (o. …
  -/
  apply nfpFamily_fp
  /-
    case H
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Ordinal.IsNormal (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x …
  -/
  rw [familyOfBFamily_enum]
  /-
    case H
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Ordinal.IsNormal (f i ?H.hi)
  -/
  exact H
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem apply_le_nfpBFamily (ho : o ≠ 0) (H : ∀ i hi, IsNormal (f i hi)) {a b} :
    (∀ i hi, f i hi b ≤ nfpBFamily.{u, v} o f a) ↔ b ≤ nfpBFamily.{u, v} o f a := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    ho : Ne o 0
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a b : Ordinal.{max u v}
    ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi b) (o.nfpBFamily f  …
  -/
  refine ⟨fun h => ?_, fun h i hi => ?_⟩
    /-
      case refine_1
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi b) (o.nfpBFamily f a)
      ⊢ LE.le b (o.nfpBFamily f a)
    -/
  · have ho' : 0 < o := Ordinal.pos_iff_ne_zero.2 ho
    /-
      case refine_1
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi b) (o.nfpBFamily f a)
      ho' : LT.lt 0 o
      ⊢ LE.le b (o.nfpBFamily f a)
    -/
    exact (H 0 ho').le_apply.trans (h 0 ho')
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : LE.le b (o.nfpBFamily f a)
      i : Ordinal.{u}
      hi : LT.lt i o
      ⊢ LE.le (f i hi b) (o.nfpBFamily f a)
    -/
  · rw [← nfpBFamily_fp (H i hi)]
    /-
      case refine_2
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      ho : Ne o 0
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a b : Ordinal.{max u v}
      h : LE.le b (o.nfpBFamily f a)
      i : Ordinal.{u}
      hi : LT.lt i o
      ⊢ LE.le (f i hi b) (f i hi (o.nfpBFamily f a))
    -/
    exact (H i hi).monotone h
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem nfpBFamily_eq_self {a} (h : ∀ i hi, f i hi a = a) : nfpBFamily.{u, v} o f a = a :=
  nfpFamily_eq_self fun _ => h _ _


set_option linter.deprecated false in
/-- A generalization of the fixed point lemma for normal functions: any family of normal functions
    has an unbounded set of common fixed points. -/
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem not_bddAbove_fp_bfamily (H : ∀ i hi, IsNormal (f i hi)) :
    ¬ BddAbove (⋂ (i) (hi), Function.fixedPoints (f i hi)) := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    ⊢ Not (BddAbove (Set.iInter fun i => Set.iInter fun hi => Function.fixedPoints …
  -/
  rw [not_bddAbove_iff]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    ⊢ ∀ (x : Ordinal.{max u v}), Exists fun y => And (Membership.mem (Set.iInter f …
  -/
  refine fun a ↦ ⟨nfpBFamily _ f (succ a), ?_, (lt_succ a).trans_le (le_nfpBFamily f _)⟩
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Membership.mem (Set.iInter fun i => Set.iInter fun hi => Function.fixedPoint …
  -/
  rw [Set.mem_iInter₂]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ ∀ (i : Ordinal.{u}) (j : LT.lt i o), Membership.mem (Function.fixedPoints (f …
  -/
  exact fun i hi ↦ nfpBFamily_fp (H i hi) _
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- A generalization of the fixed point lemma for normal functions: any family of normal functions
    has an unbounded set of common fixed points. -/
@[deprecated not_bddAbove_fp_bfamily (since := "2024-09-20")]
theorem fp_bfamily_unbounded (H : ∀ i hi, IsNormal (f i hi)) :
    (⋂ (i) (hi), Function.fixedPoints (f i hi)).Unbounded (· < ·) := fun a =>
  ⟨nfpBFamily.{u, v} _ f a, by
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a : Ordinal.{max u v}
      ⊢ Membership.mem (Set.iInter fun i => Set.iInter fun hi => Function.fixedPoint …
    -/
    rw [Set.mem_iInter₂]
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a : Ordinal.{max u v}
      ⊢ ∀ (i : Ordinal.{u}) (j : LT.lt i o), Membership.mem (Function.fixedPoints (f …
    -/
    exact fun i hi => nfpBFamily_fp (H i hi) _, (le_nfpBFamily f a).not_lt⟩
    /-
      🎉 no goals
    -/


/-- The derivative of a family of normal functions is the sequence of their common fixed points.

This is defined as `Ordinal.derivFamily` of the type-indexed family associated to `f`. -/
@[deprecated derivFamily (since := "2024-10-14")]
def derivBFamily (o : Ordinal.{u}) (f : ∀ b < o, Ordinal.{max u v} → Ordinal.{max u v}) :
    Ordinal.{max u v} → Ordinal.{max u v} :=
  derivFamily (familyOfBFamily o f)


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem derivBFamily_eq_derivFamily {o : Ordinal} (f : ∀ b < o, Ordinal → Ordinal) :
    derivBFamily.{u, v} o f = derivFamily (familyOfBFamily o f) :=
  rfl


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem isNormal_derivBFamily {o : Ordinal} (f : ∀ b < o, Ordinal → Ordinal) :
    IsNormal (derivBFamily o f) :=
  isNormal_derivFamily _


@[deprecated isNormal_derivBFamily (since := "2024-10-11")]
alias derivBFamily_isNormal := isNormal_derivBFamily


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem derivBFamily_fp {i hi} (H : IsNormal (f i hi)) (a : Ordinal) :
    f i hi (derivBFamily.{u, v} o f a) = derivBFamily.{u, v} o f a := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Eq (f i hi (o.derivBFamily f a)) (o.derivBFamily f a)
  -/
  rw [← familyOfBFamily_enum o f]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Eq (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨i, ⋯⟩) (o. …
  -/
  apply derivFamily_fp
  /-
    case H
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Ordinal.IsNormal (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x …
  -/
  rw [familyOfBFamily_enum]
  /-
    case H
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    H : Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Ordinal.IsNormal (f i ?H.hi)
  -/
  exact H
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem le_iff_derivBFamily (H : ∀ i hi, IsNormal (f i hi)) {a} :
    (∀ i hi, f i hi a ≤ a) ↔ ∃ b, derivBFamily.{u, v} o f b = a := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi a) a) (Exists fun b …
  -/
  unfold derivBFamily
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi a) a) (Exists fun b …
  -/
  rw [← le_iff_derivFamily]
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a : Ordinal.{max u v}
      ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi a) a) (∀ (i : o.toT …
    -/
  · refine ⟨fun h i => h _ _, fun h i hi => ?_⟩
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a : Ordinal.{max u v}
      h : ∀ (i : o.toType), LE.le (o.familyOfBFamily f i a) a
      i : Ordinal.{u}
      hi : LT.lt i o
      ⊢ LE.le (f i hi a) a
    -/
    rw [← familyOfBFamily_enum o f]
    /-
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a : Ordinal.{max u v}
      h : ∀ (i : o.toType), LE.le (o.familyOfBFamily f i a) a
      i : Ordinal.{u}
      hi : LT.lt i o
      ⊢ LE.le (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨i, ⋯⟩)  …
    -/
    apply h
    /-
      🎉 no goals
    -/
    /-
      case H
      o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
      H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
      a : Ordinal.{max u v}
      ⊢ ∀ (i : o.toType), Ordinal.IsNormal (o.familyOfBFamily f i)
    -/
  · exact fun _ => H _ _
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem fp_iff_derivBFamily (H : ∀ i hi, IsNormal (f i hi)) {a} :
    (∀ i hi, f i hi a = a) ↔ ∃ b, derivBFamily.{u, v} o f b = a := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), Eq (f i hi a) a) (Exists fun b => …
  -/
  rw [← le_iff_derivBFamily H]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    ⊢ Iff (∀ (i : Ordinal.{u}) (hi : LT.lt i o), Eq (f i hi a) a) (∀ (i : Ordinal. …
  -/
  refine ⟨fun h i hi => le_of_eq (h i hi), fun h i hi => ?_⟩
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi a) a
    i : Ordinal.{u}
    hi : LT.lt i o
    ⊢ Eq (f i hi a) a
  -/
  rw [← (H i hi).le_iff_eq]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max u v}
    h : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LE.le (f i hi a) a
    i : Ordinal.{u}
    hi : LT.lt i o
    ⊢ LE.le (f i hi a) a
  -/
  exact h i hi
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- For a family of normal functions, `Ordinal.derivBFamily` enumerates the common fixed points. -/
@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
theorem derivBFamily_eq_enumOrd (H : ∀ i hi, IsNormal (f i hi)) :
    derivBFamily.{u, v} o f = enumOrd (⋂ (i) (hi), Function.fixedPoints (f i hi)) := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    ⊢ Eq (o.derivBFamily f) (Ordinal.enumOrd (Set.iInter fun i => Set.iInter fun h …
  -/
  rw [eq_comm, eq_enumOrd _ (not_bddAbove_fp_bfamily H)]
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    ⊢ And (StrictMono (o.derivBFamily f)) (Eq (Set.range (o.derivBFamily f)) (Set. …
  -/
  use (isNormal_derivBFamily f).strictMono
  /-
    case right
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    ⊢ Eq (Set.range (o.derivBFamily f)) (Set.iInter fun i => Set.iInter fun hi =>  …
  -/
  rw [Set.range_eq_iff]
  /-
    case right
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    ⊢ And (∀ (a : Ordinal.{max v u}), Membership.mem (Set.iInter fun i => Set.iInt …
  -/
  refine ⟨fun a => Set.mem_iInter₂.2 fun i hi => derivBFamily_fp (H i hi) a, fun a ha => ?_⟩
  /-
    case right
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max v u}
    ha : Membership.mem (Set.iInter fun i => Set.iInter fun hi => Function.fixedPo …
    ⊢ Exists fun a_1 => Eq (o.derivBFamily f a_1) a
  -/
  rw [Set.mem_iInter₂] at ha
  /-
    case right
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v} → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Ordinal.IsNormal (f i hi)
    a : Ordinal.{max v u}
    ha : ∀ (i : Ordinal.{u}) (j : LT.lt i o), Membership.mem (Function.fixedPoints …
    ⊢ Exists fun a_1 => Eq (o.derivBFamily f a_1) a
  -/
  rwa [← fp_iff_derivBFamily H]
  /-
    🎉 no goals
  -/


/-- The next fixed point function, the least fixed point of the normal function `f`, at least `a`.

This is defined as `nfpFamily` applied to a family consisting only of `f`. -/
def nfp (f : Ordinal → Ordinal) : Ordinal → Ordinal :=
  nfpFamily fun _ : Unit => f


theorem nfp_eq_nfpFamily (f : Ordinal → Ordinal) : nfp f = nfpFamily fun _ : Unit => f :=
  rfl


theorem iSup_iterate_eq_nfp (f : Ordinal.{u} → Ordinal.{u}) (a : Ordinal.{u}) :
    ⨆ n : ℕ, f^[n] a = nfp f a := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    a : Ordinal.{u}
    ⊢ Eq (iSup fun n => Nat.iterate f n a) (Ordinal.nfp f a)
  -/
  apply le_antisymm
    /-
      case a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      ⊢ LE.le (iSup fun n => Nat.iterate f n a) (Ordinal.nfp f a)
    -/
  · rw [Ordinal.iSup_le_iff]
    /-
      case a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      ⊢ ∀ (i : Nat), LE.le (Nat.iterate f i a) (Ordinal.nfp f a)
    -/
    intro n
    /-
      case a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      n : Nat
      ⊢ LE.le (Nat.iterate f n a) (Ordinal.nfp f a)
    -/
    rw [← List.length_replicate n Unit.unit, ← List.foldr_const f a]
    /-
      case a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      n : Nat
      ⊢ LE.le (List.foldr (fun x => f) a (List.replicate n Unit.unit)) (Ordinal.nfp  …
    -/
    exact Ordinal.le_iSup _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      ⊢ LE.le (Ordinal.nfp f a) (iSup fun n => Nat.iterate f n a)
    -/
  · apply Ordinal.iSup_le
    /-
      case a.a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      ⊢ ∀ (i : List Unit), LE.le (List.foldr (fun x => f) a i) (iSup fun n => Nat.it …
    -/
    intro l
    /-
      case a.a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      l : List Unit
      ⊢ LE.le (List.foldr (fun x => f) a l) (iSup fun n => Nat.iterate f n a)
    -/
    rw [List.foldr_const f a l]
    /-
      case a.a
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      l : List Unit
      ⊢ LE.le (Nat.iterate f l.length a) (iSup fun n => Nat.iterate f n a)
    -/
    exact Ordinal.le_iSup _ _
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-27")]
theorem sup_iterate_eq_nfp (f : Ordinal.{u} → Ordinal.{u}) (a : Ordinal.{u}) :
    (sup fun n : ℕ => f^[n] a) = nfp f a := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    a : Ordinal.{u}
    ⊢ Eq (Ordinal.sup fun n => Nat.iterate f n a) (Ordinal.nfp f a)
  -/
  refine le_antisymm ?_ (sup_le fun l => ?_)
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      ⊢ LE.le (Ordinal.sup fun n => Nat.iterate f n a) (Ordinal.nfp f a)
    -/
  · rw [sup_le_iff]
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      ⊢ ∀ (i : Nat), LE.le (Nat.iterate f i a) (Ordinal.nfp f a)
    -/
    intro n
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      n : Nat
      ⊢ LE.le (Nat.iterate f n a) (Ordinal.nfp f a)
    -/
    rw [← List.length_replicate n Unit.unit, ← List.foldr_const f a]
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      n : Nat
      ⊢ LE.le (List.foldr (fun x => f) a (List.replicate n Unit.unit)) (Ordinal.nfp  …
    -/
    apply le_sup
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      l : List Unit
      ⊢ LE.le (List.foldr (fun x => f) a l) (Ordinal.sup fun n => Nat.iterate f n a)
    -/
  · rw [List.foldr_const f a l]
    /-
      case refine_2
      f : Ordinal.{u} → Ordinal.{u}
      a : Ordinal.{u}
      l : List Unit
      ⊢ LE.le (Nat.iterate f l.length a) (Ordinal.sup fun n => Nat.iterate f n a)
    -/
    exact le_sup _ _
    /-
      🎉 no goals
    -/


theorem iterate_le_nfp (f a n) : f^[n] a ≤ nfp f a := by
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    a : Ordinal.{u_1}
    n : Nat
    ⊢ LE.le (Nat.iterate f n a) (Ordinal.nfp f a)
  -/
  rw [← iSup_iterate_eq_nfp]
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    a : Ordinal.{u_1}
    n : Nat
    ⊢ LE.le (Nat.iterate f n a) (iSup fun n => Nat.iterate f n a)
  -/
  exact Ordinal.le_iSup (fun n ↦ f^[n] a) n
  /-
    🎉 no goals
  -/


theorem le_nfp (f a) : a ≤ nfp f a :=
  iterate_le_nfp f a 0


theorem lt_nfp {a b} : a < nfp f b ↔ ∃ n, a < f^[n] b := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    a b : Ordinal.{u}
    ⊢ Iff (LT.lt a (Ordinal.nfp f b)) (Exists fun n => LT.lt a (Nat.iterate f n b))
  -/
  rw [← iSup_iterate_eq_nfp]
  /-
    f : Ordinal.{u} → Ordinal.{u}
    a b : Ordinal.{u}
    ⊢ Iff (LT.lt a (iSup fun n => Nat.iterate f n b)) (Exists fun n => LT.lt a (Na …
  -/
  exact Ordinal.lt_iSup_iff
  /-
    🎉 no goals
  -/


theorem nfp_le_iff {a b} : nfp f a ≤ b ↔ ∀ n, f^[n] a ≤ b := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    a b : Ordinal.{u}
    ⊢ Iff (LE.le (Ordinal.nfp f a) b) (∀ (n : Nat), LE.le (Nat.iterate f n a) b)
  -/
  rw [← iSup_iterate_eq_nfp]
  /-
    f : Ordinal.{u} → Ordinal.{u}
    a b : Ordinal.{u}
    ⊢ Iff (LE.le (iSup fun n => Nat.iterate f n a) b) (∀ (n : Nat), LE.le (Nat.ite …
  -/
  exact Ordinal.iSup_le_iff
  /-
    🎉 no goals
  -/


theorem nfp_le {a b} : (∀ n, f^[n] a ≤ b) → nfp f a ≤ b :=
  nfp_le_iff.2


@[simp]
theorem nfp_id : nfp id = id := by
  /-
    ⊢ Eq (Ordinal.nfp id) id
  -/
  ext
  /-
    case h
    x✝ : Ordinal.{u_1}
    ⊢ Eq (Ordinal.nfp id x✝) (id x✝)
  -/
  simp_rw [← iSup_iterate_eq_nfp, iterate_id]
  /-
    case h
    x✝ : Ordinal.{u_1}
    ⊢ Eq (iSup fun n => id x✝) (id x✝)
  -/
  exact ciSup_const
  /-
    🎉 no goals
  -/


theorem nfp_monotone (hf : Monotone f) : Monotone (nfp f) :=
  nfpFamily_monotone fun _ => hf


theorem IsNormal.apply_lt_nfp {f} (H : IsNormal f) {a b} : f b < nfp f a ↔ b < nfp f a := by
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a b : Ordinal.{u_1}
    ⊢ Iff (LT.lt (f b) (Ordinal.nfp f a)) (LT.lt b (Ordinal.nfp f a))
  -/
  unfold nfp
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a b : Ordinal.{u_1}
    ⊢ Iff (LT.lt (f b) (Ordinal.nfpFamily (fun x => f) a)) (LT.lt b (Ordinal.nfpFa …
  -/
  rw [← @apply_lt_nfpFamily_iff Unit (fun _ => f) _ _ (fun _ => H) a b]
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a b : Ordinal.{u_1}
    ⊢ Iff (LT.lt (f b) (Ordinal.nfpFamily (fun x => f) a)) (Unit → LT.lt (f b) (Or …
  -/
  exact ⟨fun h _ => h, fun h => h Unit.unit⟩
  /-
    🎉 no goals
  -/


theorem IsNormal.nfp_le_apply {f} (H : IsNormal f) {a b} : nfp f a ≤ f b ↔ nfp f a ≤ b :=
  le_iff_le_iff_lt_iff_lt.2 H.apply_lt_nfp


theorem nfp_le_fp {f} (H : Monotone f) {a b} (ab : a ≤ b) (h : f b ≤ b) : nfp f a ≤ b :=
  nfpFamily_le_fp (fun _ => H) ab fun _ => h


theorem IsNormal.nfp_fp {f} (H : IsNormal f) : ∀ a, f (nfp f a) = nfp f a :=
  @nfpFamily_fp Unit (fun _ => f) _ () H


theorem IsNormal.apply_le_nfp {f} (H : IsNormal f) {a b} : f b ≤ nfp f a ↔ b ≤ nfp f a :=
                                 /-
                                   f : Ordinal.{u_1} → Ordinal.{u_1}
                                   H : Ordinal.IsNormal f
                                   a b : Ordinal.{u_1}
                                   h : LE.le b (Ordinal.nfp f a)
                                   ⊢ LE.le (f b) (Ordinal.nfp f a)
                                 -/
  ⟨H.le_apply.trans, fun h => by simpa only [H.nfp_fp] using H.le_iff.2 h⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem nfp_eq_self {f : Ordinal → Ordinal} {a} (h : f a = a) : nfp f a = a :=
  nfpFamily_eq_self fun _ => h


/-- The fixed point lemma for normal functions: any normal function has an unbounded set of
fixed points. -/
theorem not_bddAbove_fp (H : IsNormal f) : ¬ BddAbove (Function.fixedPoints f) := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    H : Ordinal.IsNormal f
    ⊢ Not (BddAbove (Function.fixedPoints f))
  -/
  convert not_bddAbove_fp_family fun _ : Unit => H
  /-
    case h.e'_1.h.e'_3
    f : Ordinal.{u} → Ordinal.{u}
    H : Ordinal.IsNormal f
    ⊢ Eq (Function.fixedPoints f) (Set.iInter fun i => Function.fixedPoints f)
  -/
  exact (Set.iInter_const _).symm
  /-
    🎉 no goals
  -/


/-- The derivative of a normal function `f` is the sequence of fixed points of `f`.

This is defined as `Ordinal.derivFamily` applied to a trivial family consisting only of `f`. -/
def deriv (f : Ordinal → Ordinal) : Ordinal → Ordinal :=
  derivFamily fun _ : Unit => f


theorem deriv_eq_derivFamily (f : Ordinal → Ordinal) : deriv f = derivFamily fun _ : Unit => f :=
  rfl


@[simp]
theorem deriv_zero_right (f) : deriv f 0 = nfp f 0 :=
  derivFamily_zero _


@[simp]
theorem deriv_succ (f o) : deriv f (succ o) = nfp f (succ (deriv f o)) :=
  derivFamily_succ _ _


theorem deriv_limit (f) {o} : IsLimit o → deriv f o = ⨆ a : {a // a < o}, deriv f a :=
  derivFamily_limit _


theorem isNormal_deriv (f) : IsNormal (deriv f) :=
  isNormal_derivFamily _


@[deprecated isNormal_deriv (since := "2024-10-11")]
alias deriv_isNormal := isNormal_deriv


theorem deriv_id_of_nfp_id {f : Ordinal → Ordinal} (h : nfp f = id) : deriv f = id :=
                                                                /-
                                                                  f : Ordinal.{u_1} → Ordinal.{u_1}
                                                                  h : Eq (Ordinal.nfp f) id
                                                                  ⊢ And (Eq (Ordinal.deriv f 0) (id 0)) (∀ (a : Ordinal.{u_1}), Eq (Ordinal.deri …
                                                                -/
  ((isNormal_deriv _).eq_iff_zero_and_succ IsNormal.refl).2 (by simp [h])
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem IsNormal.deriv_fp {f} (H : IsNormal f) : ∀ o, f (deriv f o) = deriv f o :=
  derivFamily_fp (i := ⟨⟩) H


theorem IsNormal.le_iff_deriv {f} (H : IsNormal f) {a} : f a ≤ a ↔ ∃ o, deriv f o = a := by
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a : Ordinal.{u_1}
    ⊢ Iff (LE.le (f a) a) (Exists fun o => Eq (Ordinal.deriv f o) a)
  -/
  unfold deriv
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a : Ordinal.{u_1}
    ⊢ Iff (LE.le (f a) a) (Exists fun o => Eq (Ordinal.derivFamily (fun x => f) o) …
  -/
  rw [← le_iff_derivFamily fun _ : Unit => H]
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a : Ordinal.{u_1}
    ⊢ Iff (LE.le (f a) a) (Unit → LE.le (f a) a)
  -/
  exact ⟨fun h _ => h, fun h => h Unit.unit⟩
  /-
    🎉 no goals
  -/


theorem IsNormal.fp_iff_deriv {f} (H : IsNormal f) {a} : f a = a ↔ ∃ o, deriv f o = a := by
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    H : Ordinal.IsNormal f
    a : Ordinal.{u_1}
    ⊢ Iff (Eq (f a) a) (Exists fun o => Eq (Ordinal.deriv f o) a)
  -/
  rw [← H.le_iff_eq, H.le_iff_deriv]
  /-
    🎉 no goals
  -/


/-- `Ordinal.deriv` enumerates the fixed points of a normal function. -/
theorem deriv_eq_enumOrd (H : IsNormal f) : deriv f = enumOrd (Function.fixedPoints f) := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    H : Ordinal.IsNormal f
    ⊢ Eq (Ordinal.deriv f) (Ordinal.enumOrd (Function.fixedPoints f))
  -/
  convert derivFamily_eq_enumOrd fun _ : Unit => H
  /-
    case h.e'_3.h.e'_1
    f : Ordinal.{u} → Ordinal.{u}
    H : Ordinal.IsNormal f
    ⊢ Eq (Function.fixedPoints f) (Set.iInter fun i => Function.fixedPoints f)
  -/
  exact (Set.iInter_const _).symm
  /-
    🎉 no goals
  -/


theorem deriv_eq_id_of_nfp_eq_id {f : Ordinal → Ordinal} (h : nfp f = id) : deriv f = id :=
                                                                           /-
                                                                             f : Ordinal.{u_1} → Ordinal.{u_1}
                                                                             h : Eq (Ordinal.nfp f) id
                                                                             ⊢ And (Eq (Ordinal.deriv f 0) (id 0)) (∀ (a : Ordinal.{u_1}), Eq (Ordinal.deri …
                                                                           -/
  (IsNormal.eq_iff_zero_and_succ (isNormal_deriv _) IsNormal.refl).2 <| by simp [h]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem nfp_zero_left (a) : nfp 0 a = a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (Ordinal.nfp 0 a) a
  -/
  rw [← iSup_iterate_eq_nfp]
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (iSup fun n => Nat.iterate 0 n a) a
  -/
  apply (Ordinal.iSup_le ?_).antisymm (Ordinal.le_iSup _ 0)
  /-
    a : Ordinal.{u_1}
    ⊢ ∀ (i : Nat), LE.le (Nat.iterate 0 i a) (Nat.iterate 0 0 a)
  -/
  intro n
  /-
    a : Ordinal.{u_1}
    n : Nat
    ⊢ LE.le (Nat.iterate 0 n a) (Nat.iterate 0 0 a)
  -/
  cases n
    /-
      case zero
      a : Ordinal.{u_1}
      ⊢ LE.le (Nat.iterate 0 0 a) (Nat.iterate 0 0 a)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      a : Ordinal.{u_1}
      n✝ : Nat
      ⊢ LE.le (Nat.iterate 0 (HAdd.hAdd n✝ 1) a) (Nat.iterate 0 0 a)
    -/
  · rw [Function.iterate_succ']
    /-
      case succ
      a : Ordinal.{u_1}
      n✝ : Nat
      ⊢ LE.le (Function.comp 0 (Nat.iterate 0 n✝) a) (Nat.iterate 0 0 a)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem nfp_zero : nfp 0 = id := by
  /-
    ⊢ Eq (Ordinal.nfp 0) id
  -/
  ext
  /-
    case h
    x✝ : Ordinal.{u_1}
    ⊢ Eq (Ordinal.nfp 0 x✝) (id x✝)
  -/
  exact nfp_zero_left _
  /-
    🎉 no goals
  -/


@[simp]
theorem deriv_zero : deriv 0 = id :=
  deriv_eq_id_of_nfp_eq_id nfp_zero


theorem deriv_zero_left (a) : deriv 0 a = a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (Ordinal.deriv 0 a) a
  -/
  rw [deriv_zero, id_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem nfp_add_zero (a) : nfp (a + ·) 0 = a * ω := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (Ordinal.nfp (fun x => HAdd.hAdd a x) 0) (HMul.hMul a Ordinal.omega0)
  -/
  simp_rw [← iSup_iterate_eq_nfp, ← iSup_mul_nat]
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (iSup fun n => Nat.iterate (fun x => HAdd.hAdd a x) n 0) (iSup fun n => H …
  -/
  congr; funext n
  /-
    case e_s.h
    a : Ordinal.{u_1}
    n : Nat
    ⊢ Eq (Nat.iterate (fun x => HAdd.hAdd a x) n 0) (HMul.hMul a ↑n)
  -/
  induction' n with n hn
    /-
      case e_s.h.zero
      a : Ordinal.{u_1}
      ⊢ Eq (Nat.iterate (fun x => HAdd.hAdd a x) 0 0) (HMul.hMul a ↑0)
    -/
  · rw [Nat.cast_zero, mul_zero, iterate_zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.succ
      a : Ordinal.{u_1}
      n : Nat
      hn : Eq (Nat.iterate (fun x => HAdd.hAdd a x) n 0) (HMul.hMul a ↑n)
      ⊢ Eq (Nat.iterate (fun x => HAdd.hAdd a x) (HAdd.hAdd n 1) 0) (HMul.hMul a ↑(H …
    -/
  · rw [iterate_succ_apply', Nat.add_comm, Nat.cast_add, Nat.cast_one, mul_one_add, hn]
    /-
      🎉 no goals
    -/


theorem nfp_add_eq_mul_omega0 {a b} (hba : b ≤ a * ω) : nfp (a + ·) b = a * ω := by
  /-
    a b : Ordinal.{u_1}
    hba : LE.le b (HMul.hMul a Ordinal.omega0)
    ⊢ Eq (Ordinal.nfp (fun x => HAdd.hAdd a x) b) (HMul.hMul a Ordinal.omega0)
  -/
  apply le_antisymm (nfp_le_fp (isNormal_add_right a).monotone hba _)
    /-
      a b : Ordinal.{u_1}
      hba : LE.le b (HMul.hMul a Ordinal.omega0)
      ⊢ LE.le (HMul.hMul a Ordinal.omega0) (Ordinal.nfp (fun x => HAdd.hAdd a x) b)
    -/
  · rw [← nfp_add_zero]
    /-
      a b : Ordinal.{u_1}
      hba : LE.le b (HMul.hMul a Ordinal.omega0)
      ⊢ LE.le (Ordinal.nfp (fun x => HAdd.hAdd a x) 0) (Ordinal.nfp (fun x => HAdd.h …
    -/
    exact nfp_monotone (isNormal_add_right a).monotone (Ordinal.zero_le b)
    /-
      🎉 no goals
    -/
    /-
      a b : Ordinal.{u_1}
      hba : LE.le b (HMul.hMul a Ordinal.omega0)
      ⊢ LE.le (HAdd.hAdd a (HMul.hMul a Ordinal.omega0)) (HMul.hMul a Ordinal.omega0)
    -/
  · dsimp; rw [← mul_one_add, one_add_omega0]
           /-
             🎉 no goals
           -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias nfp_add_eq_mul_omega := nfp_add_eq_mul_omega0


theorem add_eq_right_iff_mul_omega0_le {a b : Ordinal} : a + b = b ↔ a * ω ≤ b := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (Eq (HAdd.hAdd a b) b) (LE.le (HMul.hMul a Ordinal.omega0) b)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      a b : Ordinal.{u_1}
      h : Eq (HAdd.hAdd a b) b
      ⊢ LE.le (HMul.hMul a Ordinal.omega0) b
    -/
  · rw [← nfp_add_zero a, ← deriv_zero_right]
    /-
      case refine_1
      a b : Ordinal.{u_1}
      h : Eq (HAdd.hAdd a b) b
      ⊢ LE.le (Ordinal.deriv (fun x => HAdd.hAdd a x) 0) b
    -/
    cases' (isNormal_add_right a).fp_iff_deriv.1 h with c hc
    /-
      case refine_1.intro
      a b : Ordinal.{u_1}
      h : Eq (HAdd.hAdd a b) b
      c : Ordinal.{u_1}
      hc : Eq (Ordinal.deriv (fun x => HAdd.hAdd a x) c) b
      ⊢ LE.le (Ordinal.deriv (fun x => HAdd.hAdd a x) 0) b
    -/
    rw [← hc]
    /-
      case refine_1.intro
      a b : Ordinal.{u_1}
      h : Eq (HAdd.hAdd a b) b
      c : Ordinal.{u_1}
      hc : Eq (Ordinal.deriv (fun x => HAdd.hAdd a x) c) b
      ⊢ LE.le (Ordinal.deriv (fun x => HAdd.hAdd a x) 0) (Ordinal.deriv (fun x => HA …
    -/
    exact (isNormal_deriv _).monotone (Ordinal.zero_le _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Ordinal.{u_1}
      h : LE.le (HMul.hMul a Ordinal.omega0) b
      ⊢ Eq (HAdd.hAdd a b) b
    -/
  · have := Ordinal.add_sub_cancel_of_le h
    /-
      case refine_2
      a b : Ordinal.{u_1}
      h : LE.le (HMul.hMul a Ordinal.omega0) b
      this : Eq (HAdd.hAdd (HMul.hMul a Ordinal.omega0) (HSub.hSub b (HMul.hMul a Or …
      ⊢ Eq (HAdd.hAdd a b) b
    -/
    nth_rw 1 [← this]
    /-
      case refine_2
      a b : Ordinal.{u_1}
      h : LE.le (HMul.hMul a Ordinal.omega0) b
      this : Eq (HAdd.hAdd (HMul.hMul a Ordinal.omega0) (HSub.hSub b (HMul.hMul a Or …
      ⊢ Eq (HAdd.hAdd a (HAdd.hAdd (HMul.hMul a Ordinal.omega0) (HSub.hSub b (HMul.h …
    -/
    rwa [← add_assoc, ← mul_one_add, one_add_omega0]
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias add_eq_right_iff_mul_omega_le := add_eq_right_iff_mul_omega0_le


theorem add_le_right_iff_mul_omega0_le {a b : Ordinal} : a + b ≤ b ↔ a * ω ≤ b := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (LE.le (HAdd.hAdd a b) b) (LE.le (HMul.hMul a Ordinal.omega0) b)
  -/
  rw [← add_eq_right_iff_mul_omega0_le]
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (LE.le (HAdd.hAdd a b) b) (Eq (HAdd.hAdd a b) b)
  -/
  exact (isNormal_add_right a).le_iff_eq
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias add_le_right_iff_mul_omega_le := add_le_right_iff_mul_omega0_le


theorem deriv_add_eq_mul_omega0_add (a b : Ordinal.{u}) : deriv (a + ·) b = a * ω + b := by
  /-
    a b : Ordinal.{u}
    ⊢ Eq (Ordinal.deriv (fun x => HAdd.hAdd a x) b) (HAdd.hAdd (HMul.hMul a Ordina …
  -/
  revert b
  /-
    a : Ordinal.{u}
    ⊢ ∀ (b : Ordinal.{u}), Eq (Ordinal.deriv (fun x => HAdd.hAdd a x) b) (HAdd.hAd …
  -/
  rw [← funext_iff, IsNormal.eq_iff_zero_and_succ (isNormal_deriv _) (isNormal_add_right _)]
  /-
    a : Ordinal.{u}
    ⊢ And (Eq (Ordinal.deriv (fun x => HAdd.hAdd a x) 0) (HAdd.hAdd (HMul.hMul a O …
  -/
  refine ⟨?_, fun a h => ?_⟩
    /-
      case refine_1
      a : Ordinal.{u}
      ⊢ Eq (Ordinal.deriv (fun x => HAdd.hAdd a x) 0) (HAdd.hAdd (HMul.hMul a Ordina …
    -/
  · rw [deriv_zero_right, add_zero]
    /-
      case refine_1
      a : Ordinal.{u}
      ⊢ Eq (Ordinal.nfp (fun x => HAdd.hAdd a x) 0) (HMul.hMul a Ordinal.omega0)
    -/
    exact nfp_add_zero a
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a✝ a : Ordinal.{u}
      h : Eq (Ordinal.deriv (fun x => HAdd.hAdd a✝ x) a) (HAdd.hAdd (HMul.hMul a✝ Or …
      ⊢ Eq (Ordinal.deriv (fun x => HAdd.hAdd a✝ x) (Order.succ a)) (HAdd.hAdd (HMul …
    -/
  · rw [deriv_succ, h, add_succ]
    /-
      case refine_2
      a✝ a : Ordinal.{u}
      h : Eq (Ordinal.deriv (fun x => HAdd.hAdd a✝ x) a) (HAdd.hAdd (HMul.hMul a✝ Or …
      ⊢ Eq (Ordinal.nfp (fun x => HAdd.hAdd a✝ x) (Order.succ (HAdd.hAdd (HMul.hMul  …
    -/
    exact nfp_eq_self (add_eq_right_iff_mul_omega0_le.2 ((le_add_right _ _).trans (le_succ _)))
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias deriv_add_eq_mul_omega_add := deriv_add_eq_mul_omega0_add


@[simp]
theorem nfp_mul_one {a : Ordinal} (ha : 0 < a) : nfp (a * ·) 1 = a ^ ω := by
  /-
    a : Ordinal.{u_1}
    ha : LT.lt 0 a
    ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) 1) (HPow.hPow a Ordinal.omega0)
  -/
  rw [← iSup_iterate_eq_nfp, ← iSup_pow ha]
  /-
    a : Ordinal.{u_1}
    ha : LT.lt 0 a
    ⊢ Eq (iSup fun n => Nat.iterate (fun x => HMul.hMul a x) n 1) (iSup fun n => H …
  -/
  congr
  /-
    case e_s
    a : Ordinal.{u_1}
    ha : LT.lt 0 a
    ⊢ Eq (fun n => Nat.iterate (fun x => HMul.hMul a x) n 1) fun n => HPow.hPow a n
  -/
  funext n
  /-
    case e_s.h
    a : Ordinal.{u_1}
    ha : LT.lt 0 a
    n : Nat
    ⊢ Eq (Nat.iterate (fun x => HMul.hMul a x) n 1) (HPow.hPow a n)
  -/
  induction' n with n hn
    /-
      case e_s.h.zero
      a : Ordinal.{u_1}
      ha : LT.lt 0 a
      ⊢ Eq (Nat.iterate (fun x => HMul.hMul a x) 0 1) (HPow.hPow a 0)
    -/
  · rw [pow_zero, iterate_zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.succ
      a : Ordinal.{u_1}
      ha : LT.lt 0 a
      n : Nat
      hn : Eq (Nat.iterate (fun x => HMul.hMul a x) n 1) (HPow.hPow a n)
      ⊢ Eq (Nat.iterate (fun x => HMul.hMul a x) (HAdd.hAdd n 1) 1) (HPow.hPow a (HA …
    -/
  · rw [iterate_succ_apply', Nat.add_comm, pow_add, pow_one, hn]
    /-
      🎉 no goals
    -/


@[simp]
theorem nfp_mul_zero (a : Ordinal) : nfp (a * ·) 0 = 0 := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) 0) 0
  -/
  rw [← Ordinal.le_zero, nfp_le_iff]
  /-
    a : Ordinal.{u_1}
    ⊢ ∀ (n : Nat), LE.le (Nat.iterate (fun x => HMul.hMul a x) n 0) 0
  -/
  intro n
  /-
    a : Ordinal.{u_1}
    n : Nat
    ⊢ LE.le (Nat.iterate (fun x => HMul.hMul a x) n 0) 0
  -/
  induction' n with n hn; · rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    a : Ordinal.{u_1}
    n : Nat
    hn : LE.le (Nat.iterate (fun x => HMul.hMul a x) n 0) 0
    ⊢ LE.le (Nat.iterate (fun x => HMul.hMul a x) (HAdd.hAdd n 1) 0) 0
  -/
  dsimp only; rwa [iterate_succ_apply, mul_zero]
              /-
                🎉 no goals
              -/


theorem nfp_mul_eq_opow_omega0 {a b : Ordinal} (hb : 0 < b) (hba : b ≤ a ^ ω) :
    nfp (a * ·) b = a ^ ω := by
  /-
    a b : Ordinal.{u_1}
    hb : LT.lt 0 b
    hba : LE.le b (HPow.hPow a Ordinal.omega0)
    ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) b) (HPow.hPow a Ordinal.omega0)
  -/
  rcases eq_zero_or_pos a with ha | ha
    /-
      case inl
      a b : Ordinal.{u_1}
      hb : LT.lt 0 b
      hba : LE.le b (HPow.hPow a Ordinal.omega0)
      ha : Eq a 0
      ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) b) (HPow.hPow a Ordinal.omega0)
    -/
  · rw [ha, zero_opow omega0_ne_zero] at hba ⊢
    /-
      case inl
      a b : Ordinal.{u_1}
      hb : LT.lt 0 b
      hba : LE.le b 0
      ha : Eq a 0
      ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul 0 x) b) 0
    -/
    simp_rw [Ordinal.le_zero.1 hba, zero_mul]
    /-
      case inl
      a b : Ordinal.{u_1}
      hb : LT.lt 0 b
      hba : LE.le b 0
      ha : Eq a 0
      ⊢ Eq (Ordinal.nfp (fun x => 0) 0) 0
    -/
    exact nfp_zero_left 0
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Ordinal.{u_1}
    hb : LT.lt 0 b
    hba : LE.le b (HPow.hPow a Ordinal.omega0)
    ha : LT.lt 0 a
    ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) b) (HPow.hPow a Ordinal.omega0)
  -/
  apply le_antisymm
    /-
      case inr.a
      a b : Ordinal.{u_1}
      hb : LT.lt 0 b
      hba : LE.le b (HPow.hPow a Ordinal.omega0)
      ha : LT.lt 0 a
      ⊢ LE.le (Ordinal.nfp (fun x => HMul.hMul a x) b) (HPow.hPow a Ordinal.omega0)
    -/
  · apply nfp_le_fp (isNormal_mul_right ha).monotone hba
    /-
      case inr.a
      a b : Ordinal.{u_1}
      hb : LT.lt 0 b
      hba : LE.le b (HPow.hPow a Ordinal.omega0)
      ha : LT.lt 0 a
      ⊢ LE.le (HMul.hMul a (HPow.hPow a Ordinal.omega0)) (HPow.hPow a Ordinal.omega0)
    -/
    rw [← opow_one_add, one_add_omega0]
    /-
      🎉 no goals
    -/
  /-
    case inr.a
    a b : Ordinal.{u_1}
    hb : LT.lt 0 b
    hba : LE.le b (HPow.hPow a Ordinal.omega0)
    ha : LT.lt 0 a
    ⊢ LE.le (HPow.hPow a Ordinal.omega0) (Ordinal.nfp (fun x => HMul.hMul a x) b)
  -/
  rw [← nfp_mul_one ha]
  /-
    case inr.a
    a b : Ordinal.{u_1}
    hb : LT.lt 0 b
    hba : LE.le b (HPow.hPow a Ordinal.omega0)
    ha : LT.lt 0 a
    ⊢ LE.le (Ordinal.nfp (fun x => HMul.hMul a x) 1) (Ordinal.nfp (fun x => HMul.h …
  -/
  exact nfp_monotone (isNormal_mul_right ha).monotone (one_le_iff_pos.2 hb)
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias nfp_mul_eq_opow_omega := nfp_mul_eq_opow_omega0


theorem eq_zero_or_opow_omega0_le_of_mul_eq_right {a b : Ordinal} (hab : a * b = b) :
    b = 0 ∨ a ^ ω ≤ b := by
  /-
    a b : Ordinal.{u_1}
    hab : Eq (HMul.hMul a b) b
    ⊢ Or (Eq b 0) (LE.le (HPow.hPow a Ordinal.omega0) b)
  -/
  rcases eq_zero_or_pos a with ha | ha
    /-
      case inl
      a b : Ordinal.{u_1}
      hab : Eq (HMul.hMul a b) b
      ha : Eq a 0
      ⊢ Or (Eq b 0) (LE.le (HPow.hPow a Ordinal.omega0) b)
    -/
  · rw [ha, zero_opow omega0_ne_zero]
    /-
      case inl
      a b : Ordinal.{u_1}
      hab : Eq (HMul.hMul a b) b
      ha : Eq a 0
      ⊢ Or (Eq b 0) (LE.le 0 b)
    -/
    exact Or.inr (Ordinal.zero_le b)
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Ordinal.{u_1}
    hab : Eq (HMul.hMul a b) b
    ha : LT.lt 0 a
    ⊢ Or (Eq b 0) (LE.le (HPow.hPow a Ordinal.omega0) b)
  -/
  rw [or_iff_not_imp_left]
  /-
    case inr
    a b : Ordinal.{u_1}
    hab : Eq (HMul.hMul a b) b
    ha : LT.lt 0 a
    ⊢ Not (Eq b 0) → LE.le (HPow.hPow a Ordinal.omega0) b
  -/
  intro hb
  /-
    case inr
    a b : Ordinal.{u_1}
    hab : Eq (HMul.hMul a b) b
    ha : LT.lt 0 a
    hb : Not (Eq b 0)
    ⊢ LE.le (HPow.hPow a Ordinal.omega0) b
  -/
  rw [← nfp_mul_one ha]
  /-
    case inr
    a b : Ordinal.{u_1}
    hab : Eq (HMul.hMul a b) b
    ha : LT.lt 0 a
    hb : Not (Eq b 0)
    ⊢ LE.le (Ordinal.nfp (fun x => HMul.hMul a x) 1) b
  -/
  rw [← Ne, ← one_le_iff_ne_zero] at hb
  /-
    case inr
    a b : Ordinal.{u_1}
    hab : Eq (HMul.hMul a b) b
    ha : LT.lt 0 a
    hb : LE.le 1 b
    ⊢ LE.le (Ordinal.nfp (fun x => HMul.hMul a x) 1) b
  -/
  exact nfp_le_fp (isNormal_mul_right ha).monotone hb (le_of_eq hab)
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias eq_zero_or_opow_omega_le_of_mul_eq_right := eq_zero_or_opow_omega0_le_of_mul_eq_right


theorem mul_eq_right_iff_opow_omega0_dvd {a b : Ordinal} : a * b = b ↔ a ^ ω ∣ b := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (Eq (HMul.hMul a b) b) (Dvd.dvd (HPow.hPow a Ordinal.omega0) b)
  -/
  rcases eq_zero_or_pos a with ha | ha
    /-
      case inl
      a b : Ordinal.{u_1}
      ha : Eq a 0
      ⊢ Iff (Eq (HMul.hMul a b) b) (Dvd.dvd (HPow.hPow a Ordinal.omega0) b)
    -/
  · rw [ha, zero_mul, zero_opow omega0_ne_zero, zero_dvd_iff]
    /-
      case inl
      a b : Ordinal.{u_1}
      ha : Eq a 0
      ⊢ Iff (Eq 0 b) (Eq b 0)
    -/
    exact eq_comm
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Ordinal.{u_1}
    ha : LT.lt 0 a
    ⊢ Iff (Eq (HMul.hMul a b) b) (Dvd.dvd (HPow.hPow a Ordinal.omega0) b)
  -/
  refine ⟨fun hab => ?_, fun h => ?_⟩
    /-
      case inr.refine_1
      a b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hab : Eq (HMul.hMul a b) b
      ⊢ Dvd.dvd (HPow.hPow a Ordinal.omega0) b
    -/
  · rw [dvd_iff_mod_eq_zero]
    rw [← div_add_mod b (a ^ ω), mul_add, ← mul_assoc, ← opow_one_add, one_add_omega0,
      add_left_cancel_iff] at hab
    /-
      case inr.refine_1
      a b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hab : Eq (HMul.hMul a (HMod.hMod b (HPow.hPow a Ordinal.omega0))) (HMod.hMod b …
      ⊢ Eq (HMod.hMod b (HPow.hPow a Ordinal.omega0)) 0
    -/
    cases' eq_zero_or_opow_omega0_le_of_mul_eq_right hab with hab hab
      /-
        case inr.refine_1.inl
        a b : Ordinal.{u_1}
        ha : LT.lt 0 a
        hab✝ : Eq (HMul.hMul a (HMod.hMod b (HPow.hPow a Ordinal.omega0))) (HMod.hMod  …
        hab : Eq (HMod.hMod b (HPow.hPow a Ordinal.omega0)) 0
        ⊢ Eq (HMod.hMod b (HPow.hPow a Ordinal.omega0)) 0
      -/
    · exact hab
      /-
        🎉 no goals
      -/
    /-
      case inr.refine_1.inr
      a b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hab✝ : Eq (HMul.hMul a (HMod.hMod b (HPow.hPow a Ordinal.omega0))) (HMod.hMod  …
      hab : LE.le (HPow.hPow a Ordinal.omega0) (HMod.hMod b (HPow.hPow a Ordinal.ome …
      ⊢ Eq (HMod.hMod b (HPow.hPow a Ordinal.omega0)) 0
    -/
    refine (not_lt_of_le hab (mod_lt b (opow_ne_zero ω ?_))).elim
    /-
      case inr.refine_1.inr
      a b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hab✝ : Eq (HMul.hMul a (HMod.hMod b (HPow.hPow a Ordinal.omega0))) (HMod.hMod  …
      hab : LE.le (HPow.hPow a Ordinal.omega0) (HMod.hMod b (HPow.hPow a Ordinal.ome …
      ⊢ Ne a 0
    -/
    rwa [← Ordinal.pos_iff_ne_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.refine_2
    a b : Ordinal.{u_1}
    ha : LT.lt 0 a
    h : Dvd.dvd (HPow.hPow a Ordinal.omega0) b
    ⊢ Eq (HMul.hMul a b) b
  -/
  cases' h with c hc
  /-
    case inr.refine_2.intro
    a b : Ordinal.{u_1}
    ha : LT.lt 0 a
    c : Ordinal.{u_1}
    hc : Eq b (HMul.hMul (HPow.hPow a Ordinal.omega0) c)
    ⊢ Eq (HMul.hMul a b) b
  -/
  rw [hc, ← mul_assoc, ← opow_one_add, one_add_omega0]
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias mul_eq_right_iff_opow_omega_dvd := mul_eq_right_iff_opow_omega0_dvd


theorem mul_le_right_iff_opow_omega0_dvd {a b : Ordinal} (ha : 0 < a) :
    a * b ≤ b ↔ (a ^ ω) ∣ b := by
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt 0 a
    ⊢ Iff (LE.le (HMul.hMul a b) b) (Dvd.dvd (HPow.hPow a Ordinal.omega0) b)
  -/
  rw [← mul_eq_right_iff_opow_omega0_dvd]
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt 0 a
    ⊢ Iff (LE.le (HMul.hMul a b) b) (Eq (HMul.hMul a b) b)
  -/
  exact (isNormal_mul_right ha).le_iff_eq
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias mul_le_right_iff_opow_omega_dvd := mul_le_right_iff_opow_omega0_dvd


theorem nfp_mul_opow_omega0_add {a c : Ordinal} (b) (ha : 0 < a) (hc : 0 < c)
    (hca : c ≤ a ^ ω) : nfp (a * ·) (a ^ ω * b + c) = a ^ ω * succ b := by
  /-
    a c b : Ordinal.{u_1}
    ha : LT.lt 0 a
    hc : LT.lt 0 c
    hca : LE.le c (HPow.hPow a Ordinal.omega0)
    ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow a  …
  -/
  apply le_antisymm
    /-
      case a
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      ⊢ LE.le (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
    -/
  · apply nfp_le_fp (isNormal_mul_right ha).monotone
      /-
        case a.ab
        a c b : Ordinal.{u_1}
        ha : LT.lt 0 a
        hc : LT.lt 0 c
        hca : LE.le c (HPow.hPow a Ordinal.omega0)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) c) (HMul.hMul (H …
      -/
    · rw [mul_succ]
      /-
        case a.ab
        a c b : Ordinal.{u_1}
        ha : LT.lt 0 a
        hc : LT.lt 0 c
        hca : LE.le c (HPow.hPow a Ordinal.omega0)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) c) (HAdd.hAdd (H …
      -/
      apply add_le_add_left hca
      /-
        🎉 no goals
      -/
      /-
        case a.h
        a c b : Ordinal.{u_1}
        ha : LT.lt 0 a
        hc : LT.lt 0 c
        hca : LE.le c (HPow.hPow a Ordinal.omega0)
        ⊢ LE.le (HMul.hMul a (HMul.hMul (HPow.hPow a Ordinal.omega0) (Order.succ b)))  …
      -/
    · dsimp only; rw [← mul_assoc, ← opow_one_add, one_add_omega0]
                  /-
                    🎉 no goals
                  -/
  · obtain ⟨d, hd⟩ :=
      mul_eq_right_iff_opow_omega0_dvd.1 ((isNormal_mul_right ha).nfp_fp ((a ^ ω) * b + c))
    /-
      case a.intro
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      ⊢ LE.le (HMul.hMul (HPow.hPow a Ordinal.omega0) (Order.succ b)) (Ordinal.nfp ( …
    -/
    rw [hd]
    /-
      case a.intro
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      ⊢ LE.le (HMul.hMul (HPow.hPow a Ordinal.omega0) (Order.succ b)) (HMul.hMul (HP …
    -/
    apply mul_le_mul_left'
    /-
      case a.intro.bc
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      ⊢ LE.le (Order.succ b) d
    -/
    have := le_nfp (a * ·) (a ^ ω * b + c)
    /-
      case a.intro.bc
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      this : LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) c) (Ordinal …
      ⊢ LE.le (Order.succ b) d
    -/
    rw [hd] at this
    /-
      case a.intro.bc
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      this : LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) c) (HMul.hM …
      ⊢ LE.le (Order.succ b) d
    -/
    have := (add_lt_add_left hc (a ^ ω * b)).trans_le this
    /-
      case a.intro.bc
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) c) (HMul.h …
      this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) 0) (HMul.hM …
      ⊢ LE.le (Order.succ b) d
    -/
    rw [add_zero, mul_lt_mul_iff_left (opow_pos ω ha)] at this
    /-
      case a.intro.bc
      a c b : Ordinal.{u_1}
      ha : LT.lt 0 a
      hc : LT.lt 0 c
      hca : LE.le c (HPow.hPow a Ordinal.omega0)
      d : Ordinal.{u_1}
      hd : Eq (Ordinal.nfp (fun x => HMul.hMul a x) (HAdd.hAdd (HMul.hMul (HPow.hPow …
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow a Ordinal.omega0) b) c) (HMul.h …
      this : LT.lt b d
      ⊢ LE.le (Order.succ b) d
    -/
    rwa [succ_le_iff]
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias nfp_mul_opow_omega_add := nfp_mul_opow_omega0_add


theorem deriv_mul_eq_opow_omega0_mul {a : Ordinal.{u}} (ha : 0 < a) (b) :
    deriv (a * ·) b = a ^ ω * b := by
  /-
    a : Ordinal.{u}
    ha : LT.lt 0 a
    b : Ordinal.{u}
    ⊢ Eq (Ordinal.deriv (fun x => HMul.hMul a x) b) (HMul.hMul (HPow.hPow a Ordina …
  -/
  revert b
  rw [← funext_iff,
    IsNormal.eq_iff_zero_and_succ (isNormal_deriv _) (isNormal_mul_right (opow_pos ω ha))]
  /-
    a : Ordinal.{u}
    ha : LT.lt 0 a
    ⊢ And (Eq (Ordinal.deriv (fun x => HMul.hMul a x) 0) (HMul.hMul (HPow.hPow a O …
  -/
  refine ⟨?_, fun c h => ?_⟩
    /-
      case refine_1
      a : Ordinal.{u}
      ha : LT.lt 0 a
      ⊢ Eq (Ordinal.deriv (fun x => HMul.hMul a x) 0) (HMul.hMul (HPow.hPow a Ordina …
    -/
  · dsimp only; rw [deriv_zero_right, nfp_mul_zero, mul_zero]
                /-
                  🎉 no goals
                -/
    /-
      case refine_2
      a : Ordinal.{u}
      ha : LT.lt 0 a
      c : Ordinal.{u}
      h : Eq (Ordinal.deriv (fun x => HMul.hMul a x) c) (HMul.hMul (HPow.hPow a Ordi …
      ⊢ Eq (Ordinal.deriv (fun x => HMul.hMul a x) (Order.succ c)) (HMul.hMul (HPow. …
    -/
  · rw [deriv_succ, h]
    /-
      case refine_2
      a : Ordinal.{u}
      ha : LT.lt 0 a
      c : Ordinal.{u}
      h : Eq (Ordinal.deriv (fun x => HMul.hMul a x) c) (HMul.hMul (HPow.hPow a Ordi …
      ⊢ Eq (Ordinal.nfp (fun x => HMul.hMul a x) (Order.succ (HMul.hMul (HPow.hPow a …
    -/
    exact nfp_mul_opow_omega0_add c ha zero_lt_one (one_le_iff_pos.2 (opow_pos _ ha))
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias deriv_mul_eq_opow_omega_mul := deriv_mul_eq_opow_omega0_mul


