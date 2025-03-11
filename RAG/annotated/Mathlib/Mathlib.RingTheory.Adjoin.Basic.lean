@[aesop safe 20 apply (rule_sets := [SetLike])]
theorem subset_adjoin : s ⊆ adjoin R s :=
  Algebra.gc.le_u_l s


theorem adjoin_le {S : Subalgebra R A} (H : s ⊆ S) : adjoin R s ≤ S :=
  Algebra.gc.l_le H


theorem adjoin_eq_sInf : adjoin R s = sInf { p : Subalgebra R A | s ⊆ p } :=
  le_antisymm (le_sInf fun _ h => adjoin_le h) (sInf_le subset_adjoin)


theorem adjoin_le_iff {S : Subalgebra R A} : adjoin R s ≤ S ↔ s ⊆ S :=
  Algebra.gc _ _


theorem adjoin_mono (H : s ⊆ t) : adjoin R s ≤ adjoin R t :=
  Algebra.gc.monotone_l H


theorem adjoin_eq_of_le (S : Subalgebra R A) (h₁ : s ⊆ S) (h₂ : S ≤ adjoin R s) : adjoin R s = S :=
  le_antisymm (adjoin_le h₁) h₂


theorem adjoin_eq (S : Subalgebra R A) : adjoin R ↑S = S :=
  adjoin_eq_of_le _ (Set.Subset.refl _) subset_adjoin


theorem adjoin_iUnion {α : Type*} (s : α → Set A) :
    adjoin R (Set.iUnion s) = ⨆ i : α, adjoin R (s i) :=
  (@Algebra.gc R A _ _ _).l_iSup


theorem adjoin_attach_biUnion [DecidableEq A] {α : Type*} {s : Finset α} (f : s → Finset A) :
                                                                      /-
                                                                        R : Type uR
                                                                        A : Type uA
                                                                        inst✝³ : CommSemiring R
                                                                        inst✝² : Semiring A
                                                                        inst✝¹ : Algebra R A
                                                                        inst✝ : DecidableEq A
                                                                        α : Type u_1
                                                                        s : Finset α
                                                                        f : (Subtype fun x => Membership.mem s x) → Finset A
                                                                        ⊢ Eq (Algebra.adjoin R ↑(s.attach.biUnion f)) (iSup fun x => Algebra.adjoin R  …
                                                                      -/
    adjoin R (s.attach.biUnion f : Set A) = ⨆ x, adjoin R (f x) := by simp [adjoin_iUnion]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[elab_as_elim]
theorem adjoin_induction {p : (x : A) → x ∈ adjoin R s → Prop}
    (mem : ∀ (x) (hx : x ∈ s), p x (subset_adjoin hx))
    (algebraMap : ∀ r, p (algebraMap R A r) (algebraMap_mem _ r))
    (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    {x : A} (hx : x ∈ adjoin R s) : p x hx :=
  let S : Subalgebra R A :=
    { carrier := { x | ∃ hx, p x hx }
                     /-
                       R : Type uR
                       A : Type uA
                       inst✝² : CommSemiring R
                       inst✝¹ : Semiring A
                       inst✝ : Algebra R A
                       s : Set A
                       p : (x : A) → Membership.mem (Algebra.adjoin R s) x → Prop
                       mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
                       algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
                       add : ∀ (x y : A) (hx : Membership.mem (Algebra.adjoin R s) x) (hy : Membershi …
                       mul : ∀ (x y : A) (hx : Membership.mem (Algebra.adjoin R s) x) (hy : Membershi …
                       x : A
                       hx : Membership.mem (Algebra.adjoin R s) x
                       ⊢ ∀ {a b : A}, Membership.mem (setOf fun x => Exists fun hx => p x hx) a → Mem …
                     -/
      mul_mem' := by rintro _ _ ⟨_, hpx⟩ ⟨_, hpy⟩; exact ⟨_, mul _ _ _ _ hpx hpy⟩
                                                   /-
                                                     🎉 no goals
                                                   -/
                     /-
                       R : Type uR
                       A : Type uA
                       inst✝² : CommSemiring R
                       inst✝¹ : Semiring A
                       inst✝ : Algebra R A
                       s : Set A
                       p : (x : A) → Membership.mem (Algebra.adjoin R s) x → Prop
                       mem : ∀ (x : A) (hx : Membership.mem s x), p x ⋯
                       algebraMap : ∀ (r : R), p ((_root_.algebraMap R A) r) ⋯
                       add : ∀ (x y : A) (hx : Membership.mem (Algebra.adjoin R s) x) (hy : Membershi …
                       mul : ∀ (x y : A) (hx : Membership.mem (Algebra.adjoin R s) x) (hy : Membershi …
                       x : A
                       hx : Membership.mem (Algebra.adjoin R s) x
                       ⊢ ∀ {a b : A}, Membership.mem { carrier := setOf fun x => Exists fun hx => p x …
                     -/
      add_mem' := by rintro _ _ ⟨_, hpx⟩ ⟨_, hpy⟩; exact ⟨_, add _ _ _ _ hpx hpy⟩
                                                   /-
                                                     🎉 no goals
                                                   -/
      algebraMap_mem' := fun r ↦ ⟨_, algebraMap r⟩ }
  adjoin_le (S := S) (fun y hy ↦ ⟨subset_adjoin hy, mem y hy⟩) hx |>.elim fun _ ↦ _root_.id


@[deprecated adjoin_induction (since := "2024-10-10")]
alias adjoin_induction'' := adjoin_induction


/-- Induction principle for the algebra generated by a set `s`: show that `p x y` holds for any
`x y ∈ adjoin R s` given that it holds for `x y ∈ s` and that it satisfies a number of
natural properties. -/
@[elab_as_elim]
theorem adjoin_induction₂ {s : Set A} {p : (x y : A) → x ∈ adjoin R s → y ∈ adjoin R s → Prop}
    (mem_mem : ∀ (x) (y) (hx : x ∈ s) (hy : y ∈ s), p x y (subset_adjoin hx) (subset_adjoin hy))
    (algebraMap_both : ∀ r₁ r₂, p (algebraMap R A r₁) (algebraMap R A r₂) (algebraMap_mem _ r₁)
      (algebraMap_mem _ r₂))
    (algebraMap_left : ∀ (r) (x) (hx : x ∈ s), p (algebraMap R A r) x (algebraMap_mem _ r)
      (subset_adjoin hx))
    (algebraMap_right : ∀ (r) (x) (hx : x ∈ s), p x (algebraMap R A r) (subset_adjoin hx)
      (algebraMap_mem _ r))
    (add_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x + y) z (add_mem hx hy) hz)
    (add_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y + z) hx (add_mem hy hz))
    (mul_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x * y) z (mul_mem hx hy) hz)
    (mul_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y * z) hx (mul_mem hy hz))
    {x y : A} (hx : x ∈ adjoin R s) (hy : y ∈ adjoin R s) :
    p x y hx hy := by
  induction hy using adjoin_induction with
  | mem z hz => induction hx using adjoin_induction with
    | mem _ h => exact mem_mem _ _ h hz
    | algebraMap _ => exact algebraMap_left _ _ hz
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add_left _ _ _ _ _ _ h₁ h₂
  | algebraMap r =>
    induction hx using adjoin_induction with
    | mem _ h => exact algebraMap_right _ _ h
    | algebraMap _ => exact algebraMap_both _ _
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add_left _ _ _ _ _ _ h₁ h₂
  | mul _ _ _ _ h₁ h₂ => exact mul_right _ _ _ _ _ _ h₁ h₂
  | add _ _ _ _ h₁ h₂ => exact add_right _ _ _ _ _ _ h₁ h₂


/-- The difference with `Algebra.adjoin_induction` is that this acts on the subtype. -/
@[elab_as_elim, deprecated adjoin_induction (since := "2024-10-11")]
theorem adjoin_induction' {p : adjoin R s → Prop} (mem : ∀ (x) (h : x ∈ s), p ⟨x, subset_adjoin h⟩)
    (algebraMap : ∀ r, p (algebraMap R _ r)) (add : ∀ x y, p x → p y → p (x + y))
    (mul : ∀ x y, p x → p y → p (x * y)) (x : adjoin R s) : p x :=
  Subtype.recOn x fun x hx => by
    induction hx using adjoin_induction with
    | mem _ h => exact mem _ h
    | algebraMap _ => exact algebraMap _
    | mul _ _ _ _ h₁ h₂ => exact mul _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add _ _ h₁ h₂


@[simp]
theorem adjoin_adjoin_coe_preimage {s : Set A} : adjoin R (((↑) : adjoin R s → A) ⁻¹' s) = ⊤ := by
  refine eq_top_iff.2 fun ⟨x, hx⟩ ↦
      adjoin_induction (fun a ha ↦ ?_) (fun r ↦ ?_) (fun _ _ _ _ ↦ ?_) (fun _ _ _ _ ↦ ?_) hx
    /-
      case refine_1
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      x✝ : Subtype fun x => Membership.mem (Algebra.adjoin R s) x
      x : A
      hx : Membership.mem (Algebra.adjoin R s) x
      a : A
      ha : Membership.mem s a
      ⊢ Membership.mem (Algebra.adjoin R (Set.preimage Subtype.val s)) ⟨a, ⋯⟩
    -/
  · exact subset_adjoin ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      x✝ : Subtype fun x => Membership.mem (Algebra.adjoin R s) x
      x : A
      hx : Membership.mem (Algebra.adjoin R s) x
      r : R
      ⊢ Membership.mem (Algebra.adjoin R (Set.preimage Subtype.val s)) ⟨(algebraMap  …
    -/
  · exact Subalgebra.algebraMap_mem _ r
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      x✝⁴ : Subtype fun x => Membership.mem (Algebra.adjoin R s) x
      x : A
      hx : Membership.mem (Algebra.adjoin R s) x
      x✝³ x✝² : A
      x✝¹ : Membership.mem (Algebra.adjoin R s) x✝³
      x✝ : Membership.mem (Algebra.adjoin R s) x✝²
      ⊢ Membership.mem (Algebra.adjoin R (Set.preimage Subtype.val s)) ⟨x✝³, x✝¹⟩ →  …
    -/
  · exact Subalgebra.add_mem _
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      x✝⁴ : Subtype fun x => Membership.mem (Algebra.adjoin R s) x
      x : A
      hx : Membership.mem (Algebra.adjoin R s) x
      x✝³ x✝² : A
      x✝¹ : Membership.mem (Algebra.adjoin R s) x✝³
      x✝ : Membership.mem (Algebra.adjoin R s) x✝²
      ⊢ Membership.mem (Algebra.adjoin R (Set.preimage Subtype.val s)) ⟨x✝³, x✝¹⟩ →  …
    -/
  · exact Subalgebra.mul_mem _
    /-
      🎉 no goals
    -/


theorem adjoin_union (s t : Set A) : adjoin R (s ∪ t) = adjoin R s ⊔ adjoin R t :=
  (Algebra.gc : GaloisConnection _ ((↑) : Subalgebra R A → Set A)).l_sup


@[simp]
theorem adjoin_empty : adjoin R (∅ : Set A) = ⊥ :=
  show adjoin R ⊥ = ⊥ by
    /-
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ Eq (Algebra.adjoin R Bot.bot) Bot.bot
    -/
    apply GaloisConnection.l_bot
    /-
      case gc
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      ⊢ GaloisConnection (Algebra.adjoin R) ?u
    -/
    exact Algebra.gc
    /-
      🎉 no goals
    -/


@[simp]
theorem adjoin_univ : adjoin R (Set.univ : Set A) = ⊤ :=
  eq_top_iff.2 fun _x => subset_adjoin <| Set.mem_univ _


theorem adjoin_eq_span : Subalgebra.toSubmodule (adjoin R s) = span R (Submonoid.closure s) := by
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ Eq (Subalgebra.toSubmodule (Algebra.adjoin R s)) (Submodule.span R ↑(Submono …
  -/
  apply le_antisymm
    /-
      case a
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      ⊢ LE.le (Subalgebra.toSubmodule (Algebra.adjoin R s)) (Submodule.span R ↑(Subm …
    -/
  · intro r hr
    /-
      case a
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      r : A
      hr : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R s)) r
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) r
    -/
    rcases Subsemiring.mem_closure_iff_exists_list.1 hr with ⟨L, HL, rfl⟩
    /-
      case a.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      L : List (List A)
      HL : ∀ (t : List A), Membership.mem L t → ∀ (y : A), Membership.mem t y → Memb …
      hr : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R s)) (List.map Li …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) (List.map List.prod …
    -/
    clear hr
    /-
      case a.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      L : List (List A)
      HL : ∀ (t : List A), Membership.mem L t → ∀ (y : A), Membership.mem t y → Memb …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) (List.map List.prod …
    -/
    induction' L with hd tl ih
      /-
        case a.intro.intro.nil
        R : Type uR
        A : Type uA
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        s : Set A
        HL : ∀ (t : List A), Membership.mem List.nil t → ∀ (y : A), Membership.mem t y …
        ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) (List.map List.prod …
      -/
    · exact zero_mem _
      /-
        🎉 no goals
      -/
    /-
      case a.intro.intro.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : List A
      tl : List (List A)
      ih : (∀ (t : List A), Membership.mem tl t → ∀ (y : A), Membership.mem t y → Me …
      HL : ∀ (t : List A), Membership.mem (List.cons hd tl) t → ∀ (y : A), Membershi …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) (List.map List.prod …
    -/
    rw [List.forall_mem_cons] at HL
    /-
      case a.intro.intro.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : List A
      tl : List (List A)
      ih : (∀ (t : List A), Membership.mem tl t → ∀ (y : A), Membership.mem t y → Me …
      HL : And (∀ (y : A), Membership.mem hd y → Membership.mem (Union.union (Set.ra …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) (List.map List.prod …
    -/
    rw [List.map_cons, List.sum_cons]
    /-
      case a.intro.intro.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : List A
      tl : List (List A)
      ih : (∀ (t : List A), Membership.mem tl t → ∀ (y : A), Membership.mem t y → Me …
      HL : And (∀ (y : A), Membership.mem hd y → Membership.mem (Union.union (Set.ra …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) (HAdd.hAdd hd.prod  …
    -/
    refine Submodule.add_mem _ ?_ (ih HL.2)
    /-
      case a.intro.intro.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : List A
      tl : List (List A)
      ih : (∀ (t : List A), Membership.mem tl t → ∀ (y : A), Membership.mem t y → Me …
      HL : And (∀ (y : A), Membership.mem hd y → Membership.mem (Union.union (Set.ra …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) hd.prod
    -/
    replace HL := HL.1
    /-
      case a.intro.intro.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : List A
      tl : List (List A)
      ih : (∀ (t : List A), Membership.mem tl t → ∀ (y : A), Membership.mem t y → Me …
      HL : ∀ (y : A), Membership.mem hd y → Membership.mem (Union.union (Set.range ⇑ …
      ⊢ Membership.mem (Submodule.span R ↑(Submonoid.closure s)) hd.prod
    -/
    clear ih tl
    suffices ∃ (z r : _) (_hr : r ∈ Submonoid.closure s), z • r = List.prod hd by
      rcases this with ⟨z, r, hr, hzr⟩
      rw [← hzr]
      exact smul_mem _ _ (subset_span hr)
    /-
      case a.intro.intro.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : List A
      HL : ∀ (y : A), Membership.mem hd y → Membership.mem (Union.union (Set.range ⇑ …
      ⊢ Exists fun z => Exists fun r => Exists fun _hr => Eq (HSMul.hSMul z r) hd.prod
    -/
    induction' hd with hd tl ih
      /-
        case a.intro.intro.cons.nil
        R : Type uR
        A : Type uA
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        s : Set A
        HL : ∀ (y : A), Membership.mem List.nil y → Membership.mem (Union.union (Set.r …
        ⊢ Exists fun z => Exists fun r => Exists fun _hr => Eq (HSMul.hSMul z r) List. …
      -/
    · exact ⟨1, 1, (Submonoid.closure s).one_mem', one_smul _ _⟩
      /-
        🎉 no goals
      -/
    /-
      case a.intro.intro.cons.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : A
      tl : List A
      ih : (∀ (y : A), Membership.mem tl y → Membership.mem (Union.union (Set.range  …
      HL : ∀ (y : A), Membership.mem (List.cons hd tl) y → Membership.mem (Union.uni …
      ⊢ Exists fun z => Exists fun r => Exists fun _hr => Eq (HSMul.hSMul z r) (List …
    -/
    rw [List.forall_mem_cons] at HL
    /-
      case a.intro.intro.cons.cons
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : A
      tl : List A
      ih : (∀ (y : A), Membership.mem tl y → Membership.mem (Union.union (Set.range  …
      HL : And (Membership.mem (Union.union (Set.range ⇑(algebraMap R A)) s) hd) (∀  …
      ⊢ Exists fun z => Exists fun r => Exists fun _hr => Eq (HSMul.hSMul z r) (List …
    -/
    rcases ih HL.2 with ⟨z, r, hr, hzr⟩
    /-
      case a.intro.intro.cons.cons.intro.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : A
      tl : List A
      ih : (∀ (y : A), Membership.mem tl y → Membership.mem (Union.union (Set.range  …
      HL : And (Membership.mem (Union.union (Set.range ⇑(algebraMap R A)) s) hd) (∀  …
      z : R
      r : A
      hr : Membership.mem (Submonoid.closure s) r
      hzr : Eq (HSMul.hSMul z r) tl.prod
      ⊢ Exists fun z => Exists fun r => Exists fun _hr => Eq (HSMul.hSMul z r) (List …
    -/
    rw [List.prod_cons, ← hzr]
    /-
      case a.intro.intro.cons.cons.intro.intro.intro
      R : Type uR
      A : Type uA
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      s : Set A
      hd : A
      tl : List A
      ih : (∀ (y : A), Membership.mem tl y → Membership.mem (Union.union (Set.range  …
      HL : And (Membership.mem (Union.union (Set.range ⇑(algebraMap R A)) s) hd) (∀  …
      z : R
      r : A
      hr : Membership.mem (Submonoid.closure s) r
      hzr : Eq (HSMul.hSMul z r) tl.prod
      ⊢ Exists fun z_1 => Exists fun r_1 => Exists fun _hr => Eq (HSMul.hSMul z_1 r_ …
    -/
    rcases HL.1 with (⟨hd, rfl⟩ | hs)
      /-
        case a.intro.intro.cons.cons.intro.intro.intro.inl.intro
        R : Type uR
        A : Type uA
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        s : Set A
        tl : List A
        ih : (∀ (y : A), Membership.mem tl y → Membership.mem (Union.union (Set.range  …
        z : R
        r : A
        hr : Membership.mem (Submonoid.closure s) r
        hzr : Eq (HSMul.hSMul z r) tl.prod
        hd : R
        HL : And (Membership.mem (Union.union (Set.range ⇑(algebraMap R A)) s) ((algeb …
        ⊢ Exists fun z_1 => Exists fun r_1 => Exists fun _hr => Eq (HSMul.hSMul z_1 r_ …
      -/
    · refine ⟨hd * z, r, hr, ?_⟩
      /-
        case a.intro.intro.cons.cons.intro.intro.intro.inl.intro
        R : Type uR
        A : Type uA
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        s : Set A
        tl : List A
        ih : (∀ (y : A), Membership.mem tl y → Membership.mem (Union.union (Set.range  …
        z : R
        r : A
        hr : Membership.mem (Submonoid.closure s) r
        hzr : Eq (HSMul.hSMul z r) tl.prod
        hd : R
        HL : And (Membership.mem (Union.union (Set.range ⇑(algebraMap R A)) s) ((algeb …
        ⊢ Eq (HSMul.hSMul (HMul.hMul hd z) r) (HMul.hMul ((algebraMap R A) hd) (HSMul. …
      -/
      rw [Algebra.smul_def, Algebra.smul_def, (algebraMap _ _).map_mul, _root_.mul_assoc]
      /-
        🎉 no goals
      -/
    · exact
        ⟨z, hd * r, Submonoid.mul_mem _ (Submonoid.subset_closure hs) hr,
          (mul_smul_comm _ _ _).symm⟩
  /-
    case a
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ LE.le (Submodule.span R ↑(Submonoid.closure s)) (Subalgebra.toSubmodule (Alg …
  -/
  refine span_le.2 ?_
  /-
    case a
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ HasSubset.Subset ↑(Submonoid.closure s) ↑(Subalgebra.toSubmodule (Algebra.ad …
  -/
  change Submonoid.closure s ≤ (adjoin R s).toSubsemiring.toSubmonoid
  /-
    case a
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ LE.le (Submonoid.closure s) (Algebra.adjoin R s).toSubmonoid
  -/
  exact Submonoid.closure_le.2 subset_adjoin
  /-
    🎉 no goals
  -/


theorem span_le_adjoin (s : Set A) : span R s ≤ Subalgebra.toSubmodule (adjoin R s) :=
  span_le.mpr subset_adjoin


theorem adjoin_toSubmodule_le {s : Set A} {t : Submodule R A} :
    Subalgebra.toSubmodule (adjoin R s) ≤ t ↔ ↑(Submonoid.closure s) ⊆ (t : Set A) := by
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    t : Submodule R A
    ⊢ Iff (LE.le (Subalgebra.toSubmodule (Algebra.adjoin R s)) t) (HasSubset.Subse …
  -/
  rw [adjoin_eq_span, span_le]
  /-
    🎉 no goals
  -/


theorem adjoin_eq_span_of_subset {s : Set A} (hs : ↑(Submonoid.closure s) ⊆ (span R s : Set A)) :
    Subalgebra.toSubmodule (adjoin R s) = span R s :=
  le_antisymm ((adjoin_toSubmodule_le R).mpr hs) (span_le_adjoin R s)


@[simp]
theorem adjoin_span {s : Set A} : adjoin R (Submodule.span R s : Set A) = adjoin R s :=
  le_antisymm (adjoin_le (span_le_adjoin _ _)) (adjoin_mono Submodule.subset_span)


theorem adjoin_image (f : A →ₐ[R] B) (s : Set A) : adjoin R (f '' s) = (adjoin R s).map f :=
  le_antisymm (adjoin_le <| Set.image_subset _ subset_adjoin) <|
    Subalgebra.map_le.2 <| adjoin_le <| Set.image_subset_iff.1 <| by
      -- Porting note: I don't understand how this worked in Lean 3 with just `subset_adjoin`
      simp only [Set.image_id', coe_carrier_toSubmonoid, Subalgebra.coe_toSubsemiring,
        Subalgebra.coe_comap]
      /-
        R : Type uR
        A : Type uA
        B : Type uB
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        f : AlgHom R A B
        s : Set A
        ⊢ HasSubset.Subset s (Set.preimage ⇑f ↑(Algebra.adjoin R (Set.image (⇑f) s)))
      -/
      exact fun x hx => subset_adjoin ⟨x, hx, rfl⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem adjoin_insert_adjoin (x : A) : adjoin R (insert x ↑(adjoin R s)) = adjoin R (insert x s) :=
  le_antisymm
    (adjoin_le
      (Set.insert_subset_iff.mpr
        ⟨subset_adjoin (Set.mem_insert _ _), adjoin_mono (Set.subset_insert _ _)⟩))
    (Algebra.adjoin_mono (Set.insert_subset_insert Algebra.subset_adjoin))


theorem adjoin_prod_le (s : Set A) (t : Set B) :
    adjoin R (s ×ˢ t) ≤ (adjoin R s).prod (adjoin R t) :=
  adjoin_le <| Set.prod_mono subset_adjoin subset_adjoin


theorem mem_adjoin_of_map_mul {s} {x : A} {f : A →ₗ[R] B} (hf : ∀ a₁ a₂, f (a₁ * a₂) = f a₁ * f a₂)
    (h : x ∈ adjoin R s) : f x ∈ adjoin R (f '' (s ∪ {1})) := by
  induction h using adjoin_induction with
  | mem a ha => exact subset_adjoin ⟨a, ⟨Set.subset_union_left ha, rfl⟩⟩
  | algebraMap r =>
    have : f 1 ∈ adjoin R (f '' (s ∪ {1})) :=
      subset_adjoin ⟨1, ⟨Set.subset_union_right <| Set.mem_singleton 1, rfl⟩⟩
    convert Subalgebra.smul_mem (adjoin R (f '' (s ∪ {1}))) this r
    rw [algebraMap_eq_smul_one]
    exact f.map_smul _ _
  | add y z _ _ hy hz => simpa [hy, hz] using Subalgebra.add_mem _ hy hz
  | mul y z _ _ hy hz => simpa [hf, hy, hz] using Subalgebra.mul_mem _ hy hz


theorem adjoin_inl_union_inr_eq_prod (s) (t) :
    adjoin R (LinearMap.inl R A B '' (s ∪ {1}) ∪ LinearMap.inr R A B '' (t ∪ {1})) =
      (adjoin R s).prod (adjoin R t) := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    s : Set A
    t : Set B
    ⊢ Eq (Algebra.adjoin R (Union.union (Set.image (⇑(LinearMap.inl R A B)) (Union …
  -/
  apply le_antisymm
  · simp only [adjoin_le_iff, Set.insert_subset_iff, Subalgebra.zero_mem, Subalgebra.one_mem,
      subset_adjoin,-- the rest comes from `squeeze_simp`
      Set.union_subset_iff,
      LinearMap.coe_inl, Set.mk_preimage_prod_right, Set.image_subset_iff, SetLike.mem_coe,
      Set.mk_preimage_prod_left, LinearMap.coe_inr, and_self_iff, Set.union_singleton,
      Subalgebra.coe_prod]
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : Set A
      t : Set B
      ⊢ LE.le ((Algebra.adjoin R s).prod (Algebra.adjoin R t)) (Algebra.adjoin R (Un …
    -/
  · rintro ⟨a, b⟩ ⟨ha, hb⟩
    /-
      case a.mk.intro
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : Set A
      t : Set B
      a : A
      b : B
      ha : Membership.mem ↑(Algebra.adjoin R s) { fst := a, snd := b }.1
      hb : Membership.mem ↑(Algebra.adjoin R t) { fst := a, snd := b }.2
      ⊢ Membership.mem (Algebra.adjoin R (Union.union (Set.image (⇑(LinearMap.inl R  …
    -/
    let P := adjoin R (LinearMap.inl R A B '' (s ∪ {1}) ∪ LinearMap.inr R A B '' (t ∪ {1}))
    have Ha : (a, (0 : B)) ∈ adjoin R (LinearMap.inl R A B '' (s ∪ {1})) :=
      mem_adjoin_of_map_mul R LinearMap.inl_map_mul ha
    have Hb : ((0 : A), b) ∈ adjoin R (LinearMap.inr R A B '' (t ∪ {1})) :=
      mem_adjoin_of_map_mul R LinearMap.inr_map_mul hb
    /-
      case a.mk.intro
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : Set A
      t : Set B
      a : A
      b : B
      ha : Membership.mem ↑(Algebra.adjoin R s) { fst := a, snd := b }.1
      hb : Membership.mem ↑(Algebra.adjoin R t) { fst := a, snd := b }.2
      P : Subalgebra R (Prod A B) := Algebra.adjoin R (Union.union (Set.image (⇑(Lin …
      Ha : Membership.mem (Algebra.adjoin R (Set.image (⇑(LinearMap.inl R A B)) (Uni …
      Hb : Membership.mem (Algebra.adjoin R (Set.image (⇑(LinearMap.inr R A B)) (Uni …
      ⊢ Membership.mem (Algebra.adjoin R (Union.union (Set.image (⇑(LinearMap.inl R  …
    -/
    replace Ha : (a, (0 : B)) ∈ P := adjoin_mono Set.subset_union_left Ha
    /-
      case a.mk.intro
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : Set A
      t : Set B
      a : A
      b : B
      ha : Membership.mem ↑(Algebra.adjoin R s) { fst := a, snd := b }.1
      hb : Membership.mem ↑(Algebra.adjoin R t) { fst := a, snd := b }.2
      P : Subalgebra R (Prod A B) := Algebra.adjoin R (Union.union (Set.image (⇑(Lin …
      Hb : Membership.mem (Algebra.adjoin R (Set.image (⇑(LinearMap.inr R A B)) (Uni …
      Ha : Membership.mem P { fst := a, snd := 0 }
      ⊢ Membership.mem (Algebra.adjoin R (Union.union (Set.image (⇑(LinearMap.inl R  …
    -/
    replace Hb : ((0 : A), b) ∈ P := adjoin_mono Set.subset_union_right Hb
    /-
      case a.mk.intro
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      s : Set A
      t : Set B
      a : A
      b : B
      ha : Membership.mem ↑(Algebra.adjoin R s) { fst := a, snd := b }.1
      hb : Membership.mem ↑(Algebra.adjoin R t) { fst := a, snd := b }.2
      P : Subalgebra R (Prod A B) := Algebra.adjoin R (Union.union (Set.image (⇑(Lin …
      Ha : Membership.mem P { fst := a, snd := 0 }
      Hb : Membership.mem P { fst := 0, snd := b }
      ⊢ Membership.mem (Algebra.adjoin R (Union.union (Set.image (⇑(LinearMap.inl R  …
    -/
    simpa [P] using Subalgebra.add_mem _ Ha Hb
    /-
      🎉 no goals
    -/


lemma adjoin_le_centralizer_centralizer (s : Set A) :
    adjoin R s ≤ Subalgebra.centralizer R (Subalgebra.centralizer R s) :=
  adjoin_le Set.subset_centralizer_centralizer


/-- If all elements of `s : Set A` commute pairwise, then `adjoin s` is a commutative semiring. -/
abbrev adjoinCommSemiringOfComm {s : Set A} (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a) :
    CommSemiring (adjoin R s) :=
  { (adjoin R s).toSemiring with
    mul_comm := fun ⟨_, h₁⟩ ⟨_, h₂⟩ ↦
      have := adjoin_le_centralizer_centralizer R s
      Subtype.ext <| Set.centralizer_centralizer_comm_of_comm hcomm _ (this h₁) _ (this h₂) }


lemma commute_of_mem_adjoin_of_forall_mem_commute {a b : A} {s : Set A}
    (hb : b ∈ adjoin R s) (h : ∀ b ∈ s, Commute a b) :
    Commute a b := by
  induction hb using adjoin_induction with
  | mem x hx => exact h x hx
  | algebraMap r => exact commutes r a |>.symm
  | add y z _ _ hy hz => exact hy.add_right hz
  | mul y z _ _ hy hz => exact hy.mul_right hz


lemma commute_of_mem_adjoin_singleton_of_commute {a b c : A}
    (hc : c ∈ adjoin R {b}) (h : Commute a b) :
    Commute a c :=
                                                       /-
                                                         R : Type uR
                                                         A : Type uA
                                                         inst✝² : CommSemiring R
                                                         inst✝¹ : Semiring A
                                                         inst✝ : Algebra R A
                                                         a b c : A
                                                         hc : Membership.mem (Algebra.adjoin R (Singleton.singleton b)) c
                                                         h : Commute a b
                                                         ⊢ ∀ (b_1 : A), Membership.mem (Singleton.singleton b) b_1 → Commute a b_1
                                                       -/
  commute_of_mem_adjoin_of_forall_mem_commute hc <| by simpa
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma commute_of_mem_adjoin_self {a b : A} (hb : b ∈ adjoin R {a}) :
    Commute a b :=
  commute_of_mem_adjoin_singleton_of_commute hb rfl


theorem adjoin_singleton_one : adjoin R ({1} : Set A) = ⊥ :=
  eq_bot_iff.2 <| adjoin_le <| Set.singleton_subset_iff.2 <| SetLike.mem_coe.2 <| one_mem _


theorem self_mem_adjoin_singleton (x : A) : x ∈ adjoin R ({x} : Set A) :=
  Algebra.subset_adjoin (Set.mem_singleton_iff.mpr rfl)


variable (A) in
theorem adjoin_algebraMap (s : Set S) :
    adjoin R (algebraMap S A '' s) = (adjoin R s).map (IsScalarTower.toAlgHom R S A) :=
  adjoin_image R (IsScalarTower.toAlgHom R S A) s


theorem adjoin_algebraMap_image_union_eq_adjoin_adjoin (s : Set S) (t : Set A) :
    adjoin R (algebraMap S A '' s ∪ t) = (adjoin (adjoin R s) t).restrictScalars R :=
  le_antisymm
    (closure_mono <|
      Set.union_subset (Set.range_subset_iff.2 fun r => Or.inl ⟨algebraMap R (adjoin R s) r,
        (IsScalarTower.algebraMap_apply _ _ _ _).symm⟩)
        (Set.union_subset_union_left _ fun _ ⟨_x, hx, hxs⟩ => hxs ▸ ⟨⟨_, subset_adjoin hx⟩, rfl⟩))
    (closure_le.2 <|
      Set.union_subset (Set.range_subset_iff.2 fun x => adjoin_mono Set.subset_union_left <|
        Algebra.adjoin_algebraMap R A s ▸ ⟨x, x.prop, rfl⟩)
        (Set.Subset.trans Set.subset_union_right subset_adjoin))


theorem adjoin_adjoin_of_tower (s : Set A) : adjoin S (adjoin R s : Set A) = adjoin S s := by
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    s : Set A
    ⊢ Eq (Algebra.adjoin S ↑(Algebra.adjoin R s)) (Algebra.adjoin S s)
  -/
  apply le_antisymm (adjoin_le _)
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      s : Set A
      ⊢ LE.le (Algebra.adjoin S s) (Algebra.adjoin S ↑(Algebra.adjoin R s))
    -/
  · exact adjoin_mono subset_adjoin
    /-
      🎉 no goals
    -/
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      s : Set A
      ⊢ HasSubset.Subset ↑(Algebra.adjoin R s) ↑(Algebra.adjoin S s)
    -/
  · change adjoin R s ≤ (adjoin S s).restrictScalars R
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      s : Set A
      ⊢ LE.le (Algebra.adjoin R s) (Subalgebra.restrictScalars R (Algebra.adjoin S s))
    -/
    refine adjoin_le ?_
    -- Porting note: unclear why this was broken
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      s : Set A
      ⊢ HasSubset.Subset s ↑(Subalgebra.restrictScalars R (Algebra.adjoin S s))
    -/
    have : (Subalgebra.restrictScalars R (adjoin S s) : Set A) = adjoin S s := rfl
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      s : Set A
      this : Eq ↑(Subalgebra.restrictScalars R (Algebra.adjoin S s)) ↑(Algebra.adjoi …
      ⊢ HasSubset.Subset s ↑(Subalgebra.restrictScalars R (Algebra.adjoin S s))
    -/
    rw [this]
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      s : Set A
      this : Eq ↑(Subalgebra.restrictScalars R (Algebra.adjoin S s)) ↑(Algebra.adjoi …
      ⊢ HasSubset.Subset s ↑(Algebra.adjoin S s)
    -/
    exact subset_adjoin
    /-
      🎉 no goals
    -/


theorem Subalgebra.restrictScalars_adjoin {s : Set A} :
    (adjoin S s).restrictScalars R = (IsScalarTower.toAlgHom R S A).range ⊔ adjoin R s := by
  refine le_antisymm (fun _ hx ↦ adjoin_induction
    (fun x hx ↦ le_sup_right (α := Subalgebra R A) (subset_adjoin hx))
    (fun x ↦ le_sup_left (α := Subalgebra R A) ⟨x, rfl⟩)
    (fun _ _ _ _ ↦ add_mem) (fun _ _ _ _ ↦ mul_mem) <|
    (Subalgebra.mem_restrictScalars _).mp hx) (sup_le ?_ <| adjoin_le subset_adjoin)
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    s : Set A
    ⊢ LE.le (IsScalarTower.toAlgHom R S A).range (Subalgebra.restrictScalars R (Al …
  -/
  rintro _ ⟨x, rfl⟩; exact algebraMap_mem (adjoin S s) x
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem adjoin_top {A} [Semiring A] [Algebra S A] (t : Set A) :
    adjoin (⊤ : Subalgebra R S) t = (adjoin S t).restrictScalars (⊤ : Subalgebra R S) :=
  let equivTop : Subalgebra (⊤ : Subalgebra R S) A ≃o Subalgebra S A :=
    { toFun := fun s => { s with algebraMap_mem' := fun r => s.algebraMap_mem ⟨r, trivial⟩ }
      invFun := fun s => s.restrictScalars _
      left_inv := fun _ => SetLike.coe_injective rfl
      right_inv := fun _ => SetLike.coe_injective rfl
      map_rel_iff' := @fun _ _ => Iff.rfl }
  le_antisymm
    (adjoin_le <| show t ⊆ adjoin S t from subset_adjoin)
    (equivTop.symm_apply_le.mpr <|
      adjoin_le <| show t ⊆ adjoin (⊤ : Subalgebra R S) t from subset_adjoin)


theorem adjoin_union_eq_adjoin_adjoin :
    adjoin R (s ∪ t) = (adjoin (adjoin R s) t).restrictScalars R := by
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s t : Set A
    ⊢ Eq (Algebra.adjoin R (Union.union s t)) (Subalgebra.restrictScalars R (Algeb …
  -/
  simpa using adjoin_algebraMap_image_union_eq_adjoin_adjoin R s t
  /-
    🎉 no goals
  -/


theorem adjoin_union_coe_submodule :
    Subalgebra.toSubmodule (adjoin R (s ∪ t)) =
      Subalgebra.toSubmodule (adjoin R s) * Subalgebra.toSubmodule (adjoin R t) := by
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s t : Set A
    ⊢ Eq (Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t))) (HMul.hMul  …
  -/
  rw [adjoin_eq_span, adjoin_eq_span, adjoin_eq_span, span_mul_span]
  /-
    R : Type uR
    A : Type uA
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s t : Set A
    ⊢ Eq (Submodule.span R ↑(Submonoid.closure (Union.union s t))) (Submodule.span …
  -/
  congr 1 with z; simp [Submonoid.closure_union, Submonoid.mem_sup, Set.mem_mul]
                  /-
                    🎉 no goals
                  -/


theorem pow_smul_mem_of_smul_subset_of_mem_adjoin [CommSemiring B] [Algebra R B] [Algebra A B]
    [IsScalarTower R A B] (r : A) (s : Set B) (B' : Subalgebra R B) (hs : r • s ⊆ B') {x : B}
    (hx : x ∈ adjoin R s) (hr : algebraMap A B r ∈ B') : ∃ n₀ : ℕ, ∀ n ≥ n₀, r ^ n • x ∈ B' := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    x : B
    hx : Membership.mem (Algebra.adjoin R s) x
    hr : Membership.mem B' ((algebraMap A B) r)
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Membership.mem B' (HSMul.hSMul (H …
  -/
  change x ∈ Subalgebra.toSubmodule (adjoin R s) at hx
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    x : B
    hr : Membership.mem B' ((algebraMap A B) r)
    hx : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R s)) x
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Membership.mem B' (HSMul.hSMul (H …
  -/
  rw [adjoin_eq_span, Finsupp.mem_span_iff_linearCombination] at hx
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    x : B
    hr : Membership.mem B' ((algebraMap A B) r)
    hx : Exists fun l => Eq ((Finsupp.linearCombination R Subtype.val) l) x
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Membership.mem B' (HSMul.hSMul (H …
  -/
  rcases hx with ⟨l, rfl : (l.sum fun (i : Submonoid.closure s) (c : R) => c • (i : B)) = x⟩
  /-
    case intro
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Membership.mem B' (HSMul.hSMul (H …
  -/
  choose n₁ n₂ using fun x : Submonoid.closure s => Submonoid.pow_smul_mem_closure_smul r s x.prop
  /-
    case intro
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Membership.mem B' (HSMul.hSMul (H …
  -/
  use l.support.sup n₁
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    ⊢ ∀ (n : Nat), GE.ge n (l.support.sup n₁) → Membership.mem B' (HSMul.hSMul (HP …
  -/
  intro n hn
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    ⊢ Membership.mem B' (HSMul.hSMul (HPow.hPow r n) (l.sum fun i c => HSMul.hSMul …
  -/
  rw [Finsupp.smul_sum]
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    ⊢ Membership.mem B' (l.sum fun a b => HSMul.hSMul (HPow.hPow r n) (HSMul.hSMul …
  -/
  refine B'.toSubmodule.sum_mem ?_
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    ⊢ ∀ (c : ↑↑(Submonoid.closure s)), Membership.mem l.support c → Membership.mem …
  -/
  intro a ha
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    ⊢ Membership.mem (Subalgebra.toSubmodule B') ((fun a b => HSMul.hSMul (HPow.hP …
  -/
  have : n ≥ n₁ a := le_trans (Finset.le_sup ha) hn
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    this : GE.ge n (n₁ a)
    ⊢ Membership.mem (Subalgebra.toSubmodule B') ((fun a b => HSMul.hSMul (HPow.hP …
  -/
  dsimp only
  rw [← tsub_add_cancel_of_le this, pow_add, ← smul_smul, ←
    IsScalarTower.algebraMap_smul A (l a) (a : B), smul_smul (r ^ n₁ a), mul_comm, ← smul_smul,
    smul_def, map_pow, IsScalarTower.algebraMap_smul]
  /-
    case h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    this : GE.ge n (n₁ a)
    ⊢ Membership.mem (Subalgebra.toSubmodule B') (HMul.hMul (HPow.hPow ((algebraMa …
  -/
  apply Subalgebra.mul_mem _ (Subalgebra.pow_mem _ hr _) _
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    this : GE.ge n (n₁ a)
    ⊢ Membership.mem B' (HSMul.hSMul (l a) (HSMul.hSMul (HPow.hPow r (n₁ a)) ↑a))
  -/
  refine Subalgebra.smul_mem _ ?_ _
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    this : GE.ge n (n₁ a)
    ⊢ Membership.mem B' (HSMul.hSMul (HPow.hPow r (n₁ a)) ↑a)
  -/
  change _ ∈ B'.toSubmonoid
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    this : GE.ge n (n₁ a)
    ⊢ Membership.mem B'.toSubmonoid (HSMul.hSMul (HPow.hPow r (n₁ a)) ↑a)
  -/
  rw [← Submonoid.closure_eq B'.toSubmonoid]
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring B
    inst✝² : Algebra R B
    inst✝¹ : Algebra A B
    inst✝ : IsScalarTower R A B
    r : A
    s : Set B
    B' : Subalgebra R B
    hs : HasSubset.Subset (HSMul.hSMul r s) ↑B'
    hr : Membership.mem B' ((algebraMap A B) r)
    l : Finsupp (↑↑(Submonoid.closure s)) R
    n₁ : (Subtype fun x => Membership.mem (Submonoid.closure s) x) → Nat
    n₂ : ∀ (x : Subtype fun x => Membership.mem (Submonoid.closure s) x), Membersh …
    n : Nat
    hn : GE.ge n (l.support.sup n₁)
    a : ↑↑(Submonoid.closure s)
    ha : Membership.mem l.support a
    this : GE.ge n (n₁ a)
    ⊢ Membership.mem (Submonoid.closure ↑B'.toSubmonoid) (HSMul.hSMul (HPow.hPow r …
  -/
  apply Submonoid.closure_mono hs (n₂ a)
  /-
    🎉 no goals
  -/


theorem pow_smul_mem_adjoin_smul (r : R) (s : Set A) {x : A} (hx : x ∈ adjoin R s) :
    ∃ n₀ : ℕ, ∀ n ≥ n₀, r ^ n • x ∈ adjoin R (r • s) :=
  pow_smul_mem_of_smul_subset_of_mem_adjoin r s _ subset_adjoin hx (Subalgebra.algebraMap_mem _ _)


theorem mem_adjoin_iff {s : Set A} {x : A} :
    x ∈ adjoin R s ↔ x ∈ Subring.closure (Set.range (algebraMap R A) ∪ s) :=
  ⟨fun hx =>
    Subsemiring.closure_induction Subring.subset_closure (Subring.zero_mem _) (Subring.one_mem _)
      (fun _ _ _ _ => Subring.add_mem _) (fun _ _ _ _ => Subring.mul_mem _) hx,
    suffices Subring.closure (Set.range (algebraMap R A) ∪ s) ≤ (adjoin R s).toSubring
      from (show (_ : Set A) ⊆ _ from this) (a := x)
    -- Porting note: Lean doesn't seem to recognize the defeq between the order on subobjects and
    -- subsets of their coercions to sets as easily as in Lean 3
    Subring.closure_le.2 Subsemiring.subset_closure⟩


theorem adjoin_eq_ring_closure (s : Set A) :
    (adjoin R s).toSubring = Subring.closure (Set.range (algebraMap R A) ∪ s) :=
  Subring.ext fun _x => mem_adjoin_iff


/-- If all elements of `s : Set A` commute pairwise, then `adjoin R s` is a commutative
ring. -/
abbrev adjoinCommRingOfComm {s : Set A} (hcomm : ∀ a ∈ s, ∀ b ∈ s, a * b = b * a) :
    CommRing (adjoin R s) :=
  { (adjoin R s).toRing, adjoinCommSemiringOfComm R hcomm with }


theorem map_adjoin (φ : A →ₐ[R] B) (s : Set A) : (adjoin R s).map φ = adjoin R (φ '' s) :=
  (adjoin_image _ _ _).symm


@[simp]
theorem map_adjoin_singleton (e : A →ₐ[R] B) (x : A) :
    (adjoin R {x}).map e = adjoin R {e x} := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    e : AlgHom R A B
    x : A
    ⊢ Eq (Subalgebra.map e (Algebra.adjoin R (Singleton.singleton x))) (Algebra.ad …
  -/
  rw [map_adjoin, Set.image_singleton]
  /-
    🎉 no goals
  -/


theorem adjoin_le_equalizer (φ₁ φ₂ : A →ₐ[R] B) {s : Set A} (h : s.EqOn φ₁ φ₂) :
    adjoin R s ≤ equalizer φ₁ φ₂ :=
  adjoin_le h


theorem ext_of_adjoin_eq_top {s : Set A} (h : adjoin R s = ⊤) ⦃φ₁ φ₂ : A →ₐ[R] B⦄
    (hs : s.EqOn φ₁ φ₂) : φ₁ = φ₂ :=
  ext fun _x => adjoin_le_equalizer φ₁ φ₂ hs <| h.symm ▸ trivial


/-- Two algebra morphisms are equal on `Algebra.span s`iff they are equal on s -/
theorem eqOn_adjoin_iff {φ ψ : A →ₐ[R] B} {s : Set A}  :
    Set.EqOn φ ψ (adjoin R s) ↔ Set.EqOn φ ψ s := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    φ ψ : AlgHom R A B
    s : Set A
    ⊢ Iff (Set.EqOn ⇑φ ⇑ψ ↑(Algebra.adjoin R s)) (Set.EqOn (⇑φ) (⇑ψ) s)
  -/
  have (S : Set A) : S ≤ equalizer φ ψ ↔ Set.EqOn φ ψ S := Iff.rfl
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    φ ψ : AlgHom R A B
    s : Set A
    this : ∀ (S : Set A), Iff (LE.le S ↑(AlgHom.equalizer φ ψ)) (Set.EqOn (⇑φ) (⇑ψ …
    ⊢ Iff (Set.EqOn ⇑φ ⇑ψ ↑(Algebra.adjoin R s)) (Set.EqOn (⇑φ) (⇑ψ) s)
  -/
  simp only [← this, Set.le_eq_subset, SetLike.coe_subset_coe, adjoin_le_iff]
  /-
    🎉 no goals
  -/


theorem adjoin_ext {s : Set A} ⦃φ₁ φ₂ : adjoin R s →ₐ[R] B⦄
    (h : ∀ x hx, φ₁ ⟨x, subset_adjoin hx⟩ = φ₂ ⟨x, subset_adjoin hx⟩) : φ₁ = φ₂ :=
  ext fun ⟨x, hx⟩ ↦ adjoin_induction h (fun _ ↦ φ₂.commutes _ ▸ φ₁.commutes _)
                            /-
                              R : Type uR
                              A : Type uA
                              B : Type uB
                              inst✝⁴ : CommSemiring R
                              inst✝³ : Semiring A
                              inst✝² : Semiring B
                              inst✝¹ : Algebra R A
                              inst✝ : Algebra R B
                              s : Set A
                              φ₁ φ₂ : AlgHom R (Subtype fun x => Membership.mem (Algebra.adjoin R s) x) B
                              h : ∀ (x : A) (hx : Membership.mem s x), Eq (φ₁ ⟨x, ⋯⟩) (φ₂ ⟨x, ⋯⟩)
                              x✝⁴ : Subtype fun x => Membership.mem (Algebra.adjoin R s) x
                              x : A
                              hx : Membership.mem (Algebra.adjoin R s) x
                              x✝³ x✝² : A
                              x✝¹ : Membership.mem (Algebra.adjoin R s) x✝³
                              x✝ : Membership.mem (Algebra.adjoin R s) x✝²
                              h₁ : Eq (φ₁ ⟨x✝³, x✝¹⟩) (φ₂ ⟨x✝³, x✝¹⟩)
                              h₂ : Eq (φ₁ ⟨x✝², x✝⟩) (φ₂ ⟨x✝², x✝⟩)
                              ⊢ Eq (φ₁ ⟨HAdd.hAdd x✝³ x✝², ⋯⟩) (φ₂ ⟨HAdd.hAdd x✝³ x✝², ⋯⟩)
                            -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    (fun _ _ _ _ h₁ h₂ ↦ by convert congr_arg₂ (· + ·) h₁ h₂ <;> rw [← map_add] <;> rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                            /-
                              R : Type uR
                              A : Type uA
                              B : Type uB
                              inst✝⁴ : CommSemiring R
                              inst✝³ : Semiring A
                              inst✝² : Semiring B
                              inst✝¹ : Algebra R A
                              inst✝ : Algebra R B
                              s : Set A
                              φ₁ φ₂ : AlgHom R (Subtype fun x => Membership.mem (Algebra.adjoin R s) x) B
                              h : ∀ (x : A) (hx : Membership.mem s x), Eq (φ₁ ⟨x, ⋯⟩) (φ₂ ⟨x, ⋯⟩)
                              x✝⁴ : Subtype fun x => Membership.mem (Algebra.adjoin R s) x
                              x : A
                              hx : Membership.mem (Algebra.adjoin R s) x
                              x✝³ x✝² : A
                              x✝¹ : Membership.mem (Algebra.adjoin R s) x✝³
                              x✝ : Membership.mem (Algebra.adjoin R s) x✝²
                              h₁ : Eq (φ₁ ⟨x✝³, x✝¹⟩) (φ₂ ⟨x✝³, x✝¹⟩)
                              h₂ : Eq (φ₁ ⟨x✝², x✝⟩) (φ₂ ⟨x✝², x✝⟩)
                              ⊢ Eq (φ₁ ⟨HMul.hMul x✝³ x✝², ⋯⟩) (φ₂ ⟨HMul.hMul x✝³ x✝², ⋯⟩)
                            -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    (fun _ _ _ _ h₁ h₂ ↦ by convert congr_arg₂ (· * ·) h₁ h₂ <;> rw [← map_mul] <;> rfl) hx
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem ext_of_eq_adjoin {S : Subalgebra R A} {s : Set A} (hS : S = adjoin R s) ⦃φ₁ φ₂ : S →ₐ[R] B⦄
    (h : ∀ x hx, φ₁ ⟨x, hS.ge (subset_adjoin hx)⟩ = φ₂ ⟨x, hS.ge (subset_adjoin hx)⟩) :
    φ₁ = φ₂ := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    S : Subalgebra R A
    s : Set A
    hS : Eq S (Algebra.adjoin R s)
    φ₁ φ₂ : AlgHom R (Subtype fun x => Membership.mem S x) B
    h : ∀ (x : A) (hx : Membership.mem s x), Eq (φ₁ ⟨x, ⋯⟩) (φ₂ ⟨x, ⋯⟩)
    ⊢ Eq φ₁ φ₂
  -/
  subst hS; exact adjoin_ext h
            /-
              🎉 no goals
            -/


theorem Algebra.adjoin_nat {R : Type*} [Semiring R] (s : Set R) :
    adjoin ℕ s = subalgebraOfSubsemiring (Subsemiring.closure s) :=
  le_antisymm (adjoin_le Subsemiring.subset_closure)
    (Subsemiring.closure_le.2 subset_adjoin : Subsemiring.closure s ≤ (adjoin ℕ s).toSubsemiring)


theorem Algebra.adjoin_int {R : Type*} [Ring R] (s : Set R) :
    adjoin ℤ s = subalgebraOfSubring (Subring.closure s) :=
  le_antisymm (adjoin_le Subring.subset_closure)
    (Subring.closure_le.2 subset_adjoin : Subring.closure s ≤ (adjoin ℤ s).toSubring)


/-- The `ℕ`-algebra equivalence between `Subsemiring.closure s` and `Algebra.adjoin ℕ s` given
by the identity map. -/
def Subsemiring.closureEquivAdjoinNat {R : Type*} [Semiring R] (s : Set R) :
    Subsemiring.closure s ≃ₐ[ℕ] Algebra.adjoin ℕ s :=
  Subalgebra.equivOfEq (subalgebraOfSubsemiring <| Subsemiring.closure s) _ (adjoin_nat s).symm


/-- The `ℤ`-algebra equivalence between `Subring.closure s` and `Algebra.adjoin ℤ s` given by
the identity map. -/
def Subring.closureEquivAdjoinInt {R : Type*} [Ring R] (s : Set R) :
    Subring.closure s ≃ₐ[ℤ] Algebra.adjoin ℤ s :=
  Subalgebra.equivOfEq (subalgebraOfSubring <| Subring.closure s) _ (adjoin_int s).symm


/-- If `K / E / F` is a ring extension tower, `L` is a submonoid of `K / F` which is generated by
`S` as an `F`-module, then `E[L]` is generated by `S` as an `E`-module. -/
theorem Submonoid.adjoin_eq_span_of_eq_span [Semiring F] [Module F K] [IsScalarTower F E K]
    (L : Submonoid K) {S : Set K} (h : (L : Set K) = span F S) :
    toSubmodule (adjoin E (L : Set K)) = span E S := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁶ : CommSemiring E
    inst✝⁵ : Semiring K
    inst✝⁴ : SMul F E
    inst✝³ : Algebra E K
    inst✝² : Semiring F
    inst✝¹ : Module F K
    inst✝ : IsScalarTower F E K
    L : Submonoid K
    S : Set K
    h : Eq ↑L ↑(Submodule.span F S)
    ⊢ Eq (Subalgebra.toSubmodule (Algebra.adjoin E ↑L)) (Submodule.span E S)
  -/
  rw [adjoin_eq_span, L.closure_eq, h]
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁶ : CommSemiring E
    inst✝⁵ : Semiring K
    inst✝⁴ : SMul F E
    inst✝³ : Algebra E K
    inst✝² : Semiring F
    inst✝¹ : Module F K
    inst✝ : IsScalarTower F E K
    L : Submonoid K
    S : Set K
    h : Eq ↑L ↑(Submodule.span F S)
    ⊢ Eq (Submodule.span E ↑(Submodule.span F S)) (Submodule.span E S)
  -/
  exact (span_le.mpr <| span_subset_span _ _ _).antisymm (span_mono subset_span)
  /-
    🎉 no goals
  -/


/-- If `K / E / F` is a ring extension tower, `L` is a subalgebra of `K / F` which is generated by
`S` as an `F`-module, then `E[L]` is generated by `S` as an `E`-module. -/
theorem Subalgebra.adjoin_eq_span_of_eq_span {S : Set K} (h : toSubmodule L = span F S) :
    toSubmodule (adjoin E (L : Set K)) = span E S :=
  L.toSubmonoid.adjoin_eq_span_of_eq_span F E (congr_arg ((↑) : _ → Set K) h)


/-- If `K / E / F` is a ring extension tower, `L` is a subalgebra of `K / F`,
then `E[L]` is generated by any basis of `L / F` as an `E`-module. -/
theorem Subalgebra.adjoin_eq_span_basis {ι : Type*} (bL : Basis ι F L) :
    toSubmodule (adjoin E (L : Set K)) = span E (Set.range fun i : ι ↦ (bL i).1) :=
  L.adjoin_eq_span_of_eq_span E <| by
    simpa only [← L.range_val, Submodule.map_span, Submodule.map_top, ← Set.range_comp]
      using congr_arg (Submodule.map L.val) bL.span_eq.symm


theorem Algebra.restrictScalars_adjoin (F : Type*) [CommSemiring F] {E : Type*} [CommSemiring E]
    [Algebra F E] (K : Subalgebra F E) (S : Set E) :
    (Algebra.adjoin K S).restrictScalars F = Algebra.adjoin F (K ∪ S) := by
  /-
    F : Type u_4
    inst✝² : CommSemiring F
    E : Type u_5
    inst✝¹ : CommSemiring E
    inst✝ : Algebra F E
    K : Subalgebra F E
    S : Set E
    ⊢ Eq (Subalgebra.restrictScalars F (Algebra.adjoin (Subtype fun x => Membershi …
  -/
  conv_lhs => rw [← Algebra.adjoin_eq K, ← Algebra.adjoin_union_eq_adjoin_adjoin]
  /-
    🎉 no goals
  -/


/-- If `E / L / F` and `E / L' / F` are two ring extension towers, `L ≃ₐ[F] L'` is an isomorphism
compatible with `E / L` and `E / L'`, then for any subset `S` of `E`, `L[S]` and `L'[S]` are
equal as subalgebras of `E / F`. -/
theorem Algebra.restrictScalars_adjoin_of_algEquiv
    {F E L L' : Type*} [CommSemiring F] [CommSemiring L] [CommSemiring L'] [Semiring E]
    [Algebra F L] [Algebra L E] [Algebra F L'] [Algebra L' E] [Algebra F E]
    [IsScalarTower F L E] [IsScalarTower F L' E] (i : L ≃ₐ[F] L')
    (hi : algebraMap L E = (algebraMap L' E) ∘ i) (S : Set E) :
    (Algebra.adjoin L S).restrictScalars F = (Algebra.adjoin L' S).restrictScalars F := by
  /-
    F : Type u_4
    E : Type u_5
    L : Type u_6
    L' : Type u_7
    inst✝¹⁰ : CommSemiring F
    inst✝⁹ : CommSemiring L
    inst✝⁸ : CommSemiring L'
    inst✝⁷ : Semiring E
    inst✝⁶ : Algebra F L
    inst✝⁵ : Algebra L E
    inst✝⁴ : Algebra F L'
    inst✝³ : Algebra L' E
    inst✝² : Algebra F E
    inst✝¹ : IsScalarTower F L E
    inst✝ : IsScalarTower F L' E
    i : AlgEquiv F L L'
    hi : Eq (⇑(algebraMap L E)) (Function.comp ⇑(algebraMap L' E) ⇑i)
    S : Set E
    ⊢ Eq (Subalgebra.restrictScalars F (Algebra.adjoin L S)) (Subalgebra.restrictS …
  -/
  apply_fun Subalgebra.toSubsemiring using fun K K' h ↦ by rwa [SetLike.ext'_iff] at h ⊢
  /-
    F : Type u_4
    E : Type u_5
    L : Type u_6
    L' : Type u_7
    inst✝¹⁰ : CommSemiring F
    inst✝⁹ : CommSemiring L
    inst✝⁸ : CommSemiring L'
    inst✝⁷ : Semiring E
    inst✝⁶ : Algebra F L
    inst✝⁵ : Algebra L E
    inst✝⁴ : Algebra F L'
    inst✝³ : Algebra L' E
    inst✝² : Algebra F E
    inst✝¹ : IsScalarTower F L E
    inst✝ : IsScalarTower F L' E
    i : AlgEquiv F L L'
    hi : Eq (⇑(algebraMap L E)) (Function.comp ⇑(algebraMap L' E) ⇑i)
    S : Set E
    ⊢ Eq (Subalgebra.restrictScalars F (Algebra.adjoin L S)).toSubsemiring (Subalg …
  -/
  change Subsemiring.closure _ = Subsemiring.closure _
  /-
    F : Type u_4
    E : Type u_5
    L : Type u_6
    L' : Type u_7
    inst✝¹⁰ : CommSemiring F
    inst✝⁹ : CommSemiring L
    inst✝⁸ : CommSemiring L'
    inst✝⁷ : Semiring E
    inst✝⁶ : Algebra F L
    inst✝⁵ : Algebra L E
    inst✝⁴ : Algebra F L'
    inst✝³ : Algebra L' E
    inst✝² : Algebra F E
    inst✝¹ : IsScalarTower F L E
    inst✝ : IsScalarTower F L' E
    i : AlgEquiv F L L'
    hi : Eq (⇑(algebraMap L E)) (Function.comp ⇑(algebraMap L' E) ⇑i)
    S : Set E
    ⊢ Eq (Subsemiring.closure (Union.union (Set.range ⇑(algebraMap L E)) S)) (Subs …
  -/
  erw [hi, Set.range_comp, i.toEquiv.range_eq_univ, Set.image_univ]
  /-
    🎉 no goals
  -/


theorem comap_map_eq (f : A →ₐ[R] B) (S : Subalgebra R A) :
    (S.map f).comap f = S ⊔ Algebra.adjoin R (f ⁻¹' {0}) := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Ring A
    inst✝² : Algebra R A
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    f : AlgHom R A B
    S : Subalgebra R A
    ⊢ Eq (Subalgebra.comap f (Subalgebra.map f S)) (Max.max S (Algebra.adjoin R (S …
  -/
  apply le_antisymm
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      ⊢ LE.le (Subalgebra.comap f (Subalgebra.map f S)) (Max.max S (Algebra.adjoin R …
    -/
  · intro x hx
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      x : A
      hx : Membership.mem (Subalgebra.comap f (Subalgebra.map f S)) x
      ⊢ Membership.mem (Max.max S (Algebra.adjoin R (Set.preimage (⇑f) (Singleton.si …
    -/
    rw [mem_comap, mem_map] at hx
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      x : A
      hx : Exists fun x_1 => And (Membership.mem S x_1) (Eq (f x_1) (f x))
      ⊢ Membership.mem (Max.max S (Algebra.adjoin R (Set.preimage (⇑f) (Singleton.si …
    -/
    obtain ⟨y, hy, hxy⟩ := hx
    /-
      case a.intro.intro
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      x y : A
      hy : Membership.mem S y
      hxy : Eq (f y) (f x)
      ⊢ Membership.mem (Max.max S (Algebra.adjoin R (Set.preimage (⇑f) (Singleton.si …
    -/
    replace hxy : x - y ∈ f ⁻¹' {0} := by simp [hxy]
    /-
      case a.intro.intro
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      x y : A
      hy : Membership.mem S y
      hxy : Membership.mem (Set.preimage (⇑f) (Singleton.singleton 0)) (HSub.hSub x y)
      ⊢ Membership.mem (Max.max S (Algebra.adjoin R (Set.preimage (⇑f) (Singleton.si …
    -/
    rw [← Algebra.adjoin_eq S, ← Algebra.adjoin_union, ← add_sub_cancel y x]
    exact Subalgebra.add_mem _
      (Algebra.subset_adjoin <| Or.inl hy) (Algebra.subset_adjoin <| Or.inr hxy)
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      ⊢ LE.le (Max.max S (Algebra.adjoin R (Set.preimage (⇑f) (Singleton.singleton 0 …
    -/
  · rw [← map_le, Algebra.map_sup, f.map_adjoin]
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      ⊢ LE.le (Max.max (Subalgebra.map f S) (Algebra.adjoin R (Set.image (⇑f) (Set.p …
    -/
    apply le_of_eq
    /-
      case a.hab
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      ⊢ Eq (Max.max (Subalgebra.map f S) (Algebra.adjoin R (Set.image (⇑f) (Set.prei …
    -/
    rw [sup_eq_left, Algebra.adjoin_le_iff]
    /-
      case a.hab
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Ring A
      inst✝² : Algebra R A
      inst✝¹ : Ring B
      inst✝ : Algebra R B
      f : AlgHom R A B
      S : Subalgebra R A
      ⊢ HasSubset.Subset (Set.image (⇑f) (Set.preimage (⇑f) (Singleton.singleton 0)) …
    -/
    exact (Set.image_preimage_subset f {0}).trans (Set.singleton_subset_iff.2 (S.map f).zero_mem)
    /-
      🎉 no goals
    -/


theorem comap_map_eq_self {f : A →ₐ[R] B} {S : Subalgebra R A}
    (h : f ⁻¹' {0} ⊆ S) : (S.map f).comap f = S := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Ring A
    inst✝² : Algebra R A
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    f : AlgHom R A B
    S : Subalgebra R A
    h : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton 0)) ↑S
    ⊢ Eq (Subalgebra.comap f (Subalgebra.map f S)) S
  -/
  convert comap_map_eq f S
  /-
    case h.e'_3
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Ring A
    inst✝² : Algebra R A
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    f : AlgHom R A B
    S : Subalgebra R A
    h : HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton 0)) ↑S
    ⊢ Eq S (Max.max S (Algebra.adjoin R (Set.preimage (⇑f) (Singleton.singleton 0) …
  -/
  rwa [left_eq_sup, Algebra.adjoin_le_iff]
  /-
    🎉 no goals
  -/


