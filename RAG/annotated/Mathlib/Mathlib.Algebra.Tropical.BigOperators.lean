theorem List.trop_sum [AddMonoid R] (l : List R) : trop l.sum = List.prod (l.map trop) := by
  /-
    R : Type u_1
    inst✝ : AddMonoid R
    l : List R
    ⊢ Eq (Tropical.trop l.sum) (List.map Tropical.trop l).prod
  -/
  induction' l with hd tl IH
    /-
      case nil
      R : Type u_1
      inst✝ : AddMonoid R
      ⊢ Eq (Tropical.trop List.nil.sum) (List.map Tropical.trop List.nil).prod
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      inst✝ : AddMonoid R
      hd : R
      tl : List R
      IH : Eq (Tropical.trop tl.sum) (List.map Tropical.trop tl).prod
      ⊢ Eq (Tropical.trop (List.cons hd tl).sum) (List.map Tropical.trop (List.cons  …
    -/
  · simp [← IH]
    /-
      🎉 no goals
    -/


theorem Multiset.trop_sum [AddCommMonoid R] (s : Multiset R) :
    trop s.sum = Multiset.prod (s.map trop) :=
                             /-
                               R : Type u_1
                               inst✝ : AddCommMonoid R
                               s : Multiset R
                               ⊢ ∀ (a : List R), Eq (Tropical.trop (Multiset.sum (Quotient.mk (List.isSetoid  …
                             -/
  Quotient.inductionOn s (by simpa using List.trop_sum)
                             /-
                               🎉 no goals
                             -/


theorem trop_sum [AddCommMonoid R] (s : Finset S) (f : S → R) :
    trop (∑ i ∈ s, f i) = ∏ i ∈ s, trop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝ : AddCommMonoid R
    s : Finset S
    f : S → R
    ⊢ Eq (Tropical.trop (s.sum fun i => f i)) (s.prod fun i => Tropical.trop (f i))
  -/
  convert Multiset.trop_sum (s.val.map f)
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝ : AddCommMonoid R
    s : Finset S
    f : S → R
    ⊢ Eq (s.prod fun i => Tropical.trop (f i)) (Multiset.map Tropical.trop (Multis …
  -/
  simp only [Multiset.map_map, Function.comp_apply]
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝ : AddCommMonoid R
    s : Finset S
    f : S → R
    ⊢ Eq (s.prod fun i => Tropical.trop (f i)) (Multiset.map (fun i => Tropical.tr …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem List.untrop_prod [AddMonoid R] (l : List (Tropical R)) :
    untrop l.prod = List.sum (l.map untrop) := by
  /-
    R : Type u_1
    inst✝ : AddMonoid R
    l : List (Tropical R)
    ⊢ Eq (Tropical.untrop l.prod) (List.map Tropical.untrop l).sum
  -/
  induction' l with hd tl IH
    /-
      case nil
      R : Type u_1
      inst✝ : AddMonoid R
      ⊢ Eq (Tropical.untrop List.nil.prod) (List.map Tropical.untrop List.nil).sum
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      inst✝ : AddMonoid R
      hd : Tropical R
      tl : List (Tropical R)
      IH : Eq (Tropical.untrop tl.prod) (List.map Tropical.untrop tl).sum
      ⊢ Eq (Tropical.untrop (List.cons hd tl).prod) (List.map Tropical.untrop (List. …
    -/
  · simp [← IH]
    /-
      🎉 no goals
    -/


theorem Multiset.untrop_prod [AddCommMonoid R] (s : Multiset (Tropical R)) :
    untrop s.prod = Multiset.sum (s.map untrop) :=
                             /-
                               R : Type u_1
                               inst✝ : AddCommMonoid R
                               s : Multiset (Tropical R)
                               ⊢ ∀ (a : List (Tropical R)), Eq (Tropical.untrop (Multiset.prod (Quotient.mk ( …
                             -/
  Quotient.inductionOn s (by simpa using List.untrop_prod)
                             /-
                               🎉 no goals
                             -/


theorem untrop_prod [AddCommMonoid R] (s : Finset S) (f : S → Tropical R) :
    untrop (∏ i ∈ s, f i) = ∑ i ∈ s, untrop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝ : AddCommMonoid R
    s : Finset S
    f : S → Tropical R
    ⊢ Eq (Tropical.untrop (s.prod fun i => f i)) (s.sum fun i => Tropical.untrop ( …
  -/
  convert Multiset.untrop_prod (s.val.map f)
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝ : AddCommMonoid R
    s : Finset S
    f : S → Tropical R
    ⊢ Eq (s.sum fun i => Tropical.untrop (f i)) (Multiset.map Tropical.untrop (Mul …
  -/
  simp only [Multiset.map_map, Function.comp_apply]
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝ : AddCommMonoid R
    s : Finset S
    f : S → Tropical R
    ⊢ Eq (s.sum fun i => Tropical.untrop (f i)) (Multiset.map (fun i => Tropical.u …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: replaced `coe` with `WithTop.some` in statement

theorem List.trop_minimum [LinearOrder R] (l : List R) :
    trop l.minimum = List.sum (l.map (trop ∘ WithTop.some)) := by
  /-
    R : Type u_1
    inst✝ : LinearOrder R
    l : List R
    ⊢ Eq (Tropical.trop l.minimum) (List.map (Function.comp Tropical.trop WithTop. …
  -/
  induction' l with hd tl IH
    /-
      case nil
      R : Type u_1
      inst✝ : LinearOrder R
      ⊢ Eq (Tropical.trop List.nil.minimum) (List.map (Function.comp Tropical.trop W …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      inst✝ : LinearOrder R
      hd : R
      tl : List R
      IH : Eq (Tropical.trop tl.minimum) (List.map (Function.comp Tropical.trop With …
      ⊢ Eq (Tropical.trop (List.cons hd tl).minimum) (List.map (Function.comp Tropic …
    -/
  · simp [List.minimum_cons, ← IH]
    /-
      🎉 no goals
    -/


theorem Multiset.trop_inf [LinearOrder R] [OrderTop R] (s : Multiset R) :
    trop s.inf = Multiset.sum (s.map trop) := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Multiset R
    ⊢ Eq (Tropical.trop s.inf) (Multiset.map Tropical.trop s).sum
  -/
  induction' s using Multiset.induction with s x IH
    /-
      case empty
      R : Type u_1
      inst✝¹ : LinearOrder R
      inst✝ : OrderTop R
      ⊢ Eq (Tropical.trop (Multiset.inf 0)) (Multiset.map Tropical.trop 0).sum
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      inst✝¹ : LinearOrder R
      inst✝ : OrderTop R
      s : R
      x : Multiset R
      IH : Eq (Tropical.trop x.inf) (Multiset.map Tropical.trop x).sum
      ⊢ Eq (Tropical.trop (Multiset.cons s x).inf) (Multiset.map Tropical.trop (Mult …
    -/
  · simp [← IH]
    /-
      🎉 no goals
    -/


theorem Finset.trop_inf [LinearOrder R] [OrderTop R] (s : Finset S) (f : S → R) :
    trop (s.inf f) = ∑ i ∈ s, trop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Finset S
    f : S → R
    ⊢ Eq (Tropical.trop (s.inf f)) (s.sum fun i => Tropical.trop (f i))
  -/
  convert Multiset.trop_inf (s.val.map f)
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Finset S
    f : S → R
    ⊢ Eq (s.sum fun i => Tropical.trop (f i)) (Multiset.map Tropical.trop (Multise …
  -/
  simp only [Multiset.map_map, Function.comp_apply]
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Finset S
    f : S → R
    ⊢ Eq (s.sum fun i => Tropical.trop (f i)) (Multiset.map (fun i => Tropical.tro …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem trop_sInf_image [ConditionallyCompleteLinearOrder R] (s : Finset S) (f : S → WithTop R) :
    trop (sInf (f '' s)) = ∑ i ∈ s, trop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder R
    s : Finset S
    f : S → WithTop R
    ⊢ Eq (Tropical.trop (InfSet.sInf (Set.image f ↑s))) (s.sum fun i => Tropical.t …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | h)
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder R
      f : S → WithTop R
      ⊢ Eq (Tropical.trop (InfSet.sInf (Set.image f ↑EmptyCollection.emptyCollection …
    -/
  · simp only [Set.image_empty, coe_empty, sum_empty, WithTop.sInf_empty, trop_top]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder R
    s : Finset S
    f : S → WithTop R
    h : s.Nonempty
    ⊢ Eq (Tropical.trop (InfSet.sInf (Set.image f ↑s))) (s.sum fun i => Tropical.t …
  -/
  rw [← inf'_eq_csInf_image _ h, inf'_eq_inf, s.trop_inf]
  /-
    🎉 no goals
  -/


theorem trop_iInf [ConditionallyCompleteLinearOrder R] [Fintype S] (f : S → WithTop R) :
    trop (⨅ i : S, f i) = ∑ i : S, trop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : ConditionallyCompleteLinearOrder R
    inst✝ : Fintype S
    f : S → WithTop R
    ⊢ Eq (Tropical.trop (iInf fun i => f i)) (Finset.univ.sum fun i => Tropical.tr …
  -/
  rw [iInf, ← Set.image_univ, ← coe_univ, trop_sInf_image]
  /-
    🎉 no goals
  -/


theorem Multiset.untrop_sum [LinearOrder R] [OrderTop R] (s : Multiset (Tropical R)) :
    untrop s.sum = Multiset.inf (s.map untrop) := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Multiset (Tropical R)
    ⊢ Eq (Tropical.untrop s.sum) (Multiset.map Tropical.untrop s).inf
  -/
  induction' s using Multiset.induction with s x IH
    /-
      case empty
      R : Type u_1
      inst✝¹ : LinearOrder R
      inst✝ : OrderTop R
      ⊢ Eq (Tropical.untrop (Multiset.sum 0)) (Multiset.map Tropical.untrop 0).inf
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_1
      inst✝¹ : LinearOrder R
      inst✝ : OrderTop R
      s : Tropical R
      x : Multiset (Tropical R)
      IH : Eq (Tropical.untrop x.sum) (Multiset.map Tropical.untrop x).inf
      ⊢ Eq (Tropical.untrop (Multiset.cons s x).sum) (Multiset.map Tropical.untrop ( …
    -/
  · simp only [sum_cons, untrop_add, untrop_le_iff, map_cons, inf_cons, ← IH]
    /-
      🎉 no goals
    -/


theorem Finset.untrop_sum' [LinearOrder R] [OrderTop R] (s : Finset S) (f : S → Tropical R) :
    untrop (∑ i ∈ s, f i) = s.inf (untrop ∘ f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Finset S
    f : S → Tropical R
    ⊢ Eq (Tropical.untrop (s.sum fun i => f i)) (s.inf (Function.comp Tropical.unt …
  -/
  convert Multiset.untrop_sum (s.val.map f)
  /-
    case h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝¹ : LinearOrder R
    inst✝ : OrderTop R
    s : Finset S
    f : S → Tropical R
    ⊢ Eq (s.inf (Function.comp Tropical.untrop f)) (Multiset.map Tropical.untrop ( …
  -/
  simp only [Multiset.map_map, Function.comp_apply, inf_def]
  /-
    🎉 no goals
  -/


theorem untrop_sum_eq_sInf_image [ConditionallyCompleteLinearOrder R] (s : Finset S)
    (f : S → Tropical (WithTop R)) : untrop (∑ i ∈ s, f i) = sInf (untrop ∘ f '' s) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder R
    s : Finset S
    f : S → Tropical (WithTop R)
    ⊢ Eq (Tropical.untrop (s.sum fun i => f i)) (InfSet.sInf (Set.image (Function. …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | h)
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder R
      f : S → Tropical (WithTop R)
      ⊢ Eq (Tropical.untrop (EmptyCollection.emptyCollection.sum fun i => f i)) (Inf …
    -/
  · simp only [Set.image_empty, coe_empty, sum_empty, WithTop.sInf_empty, untrop_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      S : Type u_2
      inst✝ : ConditionallyCompleteLinearOrder R
      s : Finset S
      f : S → Tropical (WithTop R)
      h : s.Nonempty
      ⊢ Eq (Tropical.untrop (s.sum fun i => f i)) (InfSet.sInf (Set.image (Function. …
    -/
  · rw [← inf'_eq_csInf_image _ h, inf'_eq_inf, Finset.untrop_sum']
    /-
      🎉 no goals
    -/


theorem untrop_sum [ConditionallyCompleteLinearOrder R] [Fintype S] (f : S → Tropical (WithTop R)) :
    untrop (∑ i : S, f i) = ⨅ i : S, untrop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : ConditionallyCompleteLinearOrder R
    inst✝ : Fintype S
    f : S → Tropical (WithTop R)
    ⊢ Eq (Tropical.untrop (Finset.univ.sum fun i => f i)) (iInf fun i => Tropical. …
  -/
  rw [iInf, ← Set.image_univ, ← coe_univ, untrop_sum_eq_sInf_image, Function.comp_def]
  /-
    🎉 no goals
  -/


/-- Note we cannot use `i ∈ s` instead of `i : s` here
as it is simply not true on conditionally complete lattices! -/
theorem Finset.untrop_sum [ConditionallyCompleteLinearOrder R] (s : Finset S)
    (f : S → Tropical (WithTop R)) : untrop (∑ i ∈ s, f i) = ⨅ i : s, untrop (f i) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝ : ConditionallyCompleteLinearOrder R
    s : Finset S
    f : S → Tropical (WithTop R)
    ⊢ Eq (Tropical.untrop (s.sum fun i => f i)) (iInf fun i => Tropical.untrop (f  …
  -/
  simpa [← _root_.untrop_sum] using (sum_attach _ _).symm
  /-
    🎉 no goals
  -/

