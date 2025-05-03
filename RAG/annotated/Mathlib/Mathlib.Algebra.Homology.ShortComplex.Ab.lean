@[simp]
lemma ab_zero_apply (x : S.X₁) : S.g (S.f x) = 0 := by
  /-
    S : CategoryTheory.ShortComplex Ab
    x : ↑S.X₁
    ⊢ Eq (S.g (S.f x)) 0
  -/
  rw [← comp_apply, S.zero]
  /-
    S : CategoryTheory.ShortComplex Ab
    x : ↑S.X₁
    ⊢ Eq (0 x) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The canonical additive morphism `S.X₁ →+ AddMonoidHom.ker S.g` induced by `S.f`. -/
@[simps!]
def abToCycles : S.X₁ →+ AddMonoidHom.ker S.g :=
                                                               /-
                                                                 S : CategoryTheory.ShortComplex Ab
                                                                 ⊢ ∀ (a b : ↑S.X₁), Eq ((fun x => ⟨S.f x, ⋯⟩) (HAdd.hAdd a b)) (HAdd.hAdd ((fun …
                                                               -/
    AddMonoidHom.mk' (fun x => ⟨S.f x, S.ab_zero_apply x⟩) (by aesop)
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The explicit left homology data of a short complex of abelian group that is
given by a kernel and a quotient given by the `AddMonoidHom` API. -/
@[simps]
def abLeftHomologyData : S.LeftHomologyData where
  K := AddCommGrp.of (AddMonoidHom.ker S.g)
  H := AddCommGrp.of ((AddMonoidHom.ker S.g) ⧸ AddMonoidHom.range S.abToCycles)
  i := (AddMonoidHom.ker S.g).subtype
  π := QuotientAddGroup.mk' _
  wi := by
    /-
      S : CategoryTheory.ShortComplex Ab
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddMonoidHom.ker S.g).subtype S.g) 0
    -/
    ext ⟨_, hx⟩
    /-
      case w.mk
      S : CategoryTheory.ShortComplex Ab
      val✝ : ↑S.X₂
      hx : Membership.mem (AddMonoidHom.ker S.g) val✝
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddMonoidHom.ker S.g).subtype S.g)  …
    -/
    exact hx
    /-
      🎉 no goals
    -/
  hi := AddCommGrp.kernelIsLimit _
  wπ := by
    /-
      S : CategoryTheory.ShortComplex Ab
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AddCommGrp.kernelIsLimit S.g).lift  …
    -/
    ext (x : S.X₁)
    /-
      case w
      S : CategoryTheory.ShortComplex Ab
      x : ↑S.X₁
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AddCommGrp.kernelIsLimit S.g).lift …
    -/
    erw [QuotientAddGroup.eq_zero_iff]
    /-
      case w
      S : CategoryTheory.ShortComplex Ab
      x : ↑S.X₁
      ⊢ Membership.mem S.abToCycles.range (((AddCommGrp.kernelIsLimit S.g).lift (Cat …
    -/
    rw [AddMonoidHom.mem_range]
    /-
      case w
      S : CategoryTheory.ShortComplex Ab
      x : ↑S.X₁
      ⊢ Exists fun x_1 => Eq (S.abToCycles x_1) (((AddCommGrp.kernelIsLimit S.g).lif …
    -/
    apply exists_apply_eq_apply
    /-
      🎉 no goals
    -/
  hπ := AddCommGrp.cokernelIsColimit (AddCommGrp.ofHom S.abToCycles)


@[simp]
lemma abLeftHomologyData_f' : S.abLeftHomologyData.f' = S.abToCycles := rfl


/-- Given a short complex `S` of abelian groups, this is the isomorphism between
the abstract `S.cycles` of the homology API and the more concrete description as
`AddMonoidHom.ker S.g`. -/
noncomputable def abCyclesIso : S.cycles ≅ AddCommGrp.of (AddMonoidHom.ker S.g) :=
  S.abLeftHomologyData.cyclesIso

-- This was a simp lemma until we made `AddCommGrp.coe_of` a simp lemma,
-- after which the simp normal form linter complains.
-- It was not used a simp lemma in Mathlib.
-- Possible solution: higher priority function coercions that remove the `of`?
-- @[simp]

lemma abCyclesIso_inv_apply_iCycles (x : AddMonoidHom.ker S.g) :
    S.iCycles (S.abCyclesIso.inv x) = x := by
  /-
    S : CategoryTheory.ShortComplex Ab
    x : Subtype fun x => Membership.mem (AddMonoidHom.ker S.g) x
    ⊢ Eq (S.iCycles (S.abCyclesIso.inv x)) ↑x
  -/
  dsimp only [abCyclesIso]
  /-
    S : CategoryTheory.ShortComplex Ab
    x : Subtype fun x => Membership.mem (AddMonoidHom.ker S.g) x
    ⊢ Eq (S.iCycles (S.abLeftHomologyData.cyclesIso.inv x)) ↑x
  -/
  rw [← comp_apply, S.abLeftHomologyData.cyclesIso_inv_comp_iCycles]
  /-
    S : CategoryTheory.ShortComplex Ab
    x : Subtype fun x => Membership.mem (AddMonoidHom.ker S.g) x
    ⊢ Eq (S.abLeftHomologyData.i x) ↑x
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a short complex `S` of abelian groups, this is the isomorphism between
the abstract `S.homology` of the homology API and the more explicit
quotient of `AddMonoidHom.ker S.g` by the image of
`S.abToCycles : S.X₁ →+ AddMonoidHom.ker S.g`. -/
noncomputable def abHomologyIso : S.homology ≅
    AddCommGrp.of ((AddMonoidHom.ker S.g) ⧸ AddMonoidHom.range S.abToCycles) :=
  S.abLeftHomologyData.homologyIso


lemma exact_iff_surjective_abToCycles :
    S.Exact ↔ Function.Surjective S.abToCycles := by
  rw [S.abLeftHomologyData.exact_iff_epi_f', abLeftHomologyData_f',
    AddCommGrp.epi_iff_surjective]
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff (Function.Surjective ⇑S.abToCycles) (Function.Surjective ⇑S.abToCycles)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma ab_exact_iff :
    S.Exact ↔ ∀ (x₂ : S.X₂) (_ : S.g x₂ = 0), ∃ (x₁ : S.X₁), S.f x₁ = x₂ := by
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff S.Exact (∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂)
  -/
  rw [exact_iff_surjective_abToCycles]
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff (Function.Surjective ⇑S.abToCycles) (∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exi …
  -/
  constructor
    /-
      case mp
      S : CategoryTheory.ShortComplex Ab
      ⊢ Function.Surjective ⇑S.abToCycles → ∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exists f …
    -/
  · intro h x₂ hx₂
    /-
      case mp
      S : CategoryTheory.ShortComplex Ab
      h : Function.Surjective ⇑S.abToCycles
      x₂ : ↑S.X₂
      hx₂ : Eq (S.g x₂) 0
      ⊢ Exists fun x₁ => Eq (S.f x₁) x₂
    -/
    obtain ⟨x₁, hx₁⟩ := h ⟨x₂, hx₂⟩
    /-
      case mp.intro
      S : CategoryTheory.ShortComplex Ab
      h : Function.Surjective ⇑S.abToCycles
      x₂ : ↑S.X₂
      hx₂ : Eq (S.g x₂) 0
      x₁ : ↑S.X₁
      hx₁ : Eq (S.abToCycles x₁) ⟨x₂, hx₂⟩
      ⊢ Exists fun x₁ => Eq (S.f x₁) x₂
    -/
    exact ⟨x₁, by simpa only [Subtype.ext_iff, abToCycles_apply_coe] using hx₁⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S : CategoryTheory.ShortComplex Ab
      ⊢ (∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂) → Function …
    -/
  · rintro h ⟨x₂, hx₂⟩
    /-
      case mpr.mk
      S : CategoryTheory.ShortComplex Ab
      h : ∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂
      x₂ : ↑S.X₂
      hx₂ : Membership.mem (AddMonoidHom.ker S.g) x₂
      ⊢ Exists fun a => Eq (S.abToCycles a) ⟨x₂, hx₂⟩
    -/
    obtain ⟨x₁, rfl⟩ := h x₂ hx₂
    /-
      case mpr.mk.intro
      S : CategoryTheory.ShortComplex Ab
      h : ∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂
      x₁ : ↑S.X₁
      hx₂ : Membership.mem (AddMonoidHom.ker S.g) (S.f x₁)
      ⊢ Exists fun a => Eq (S.abToCycles a) ⟨S.f x₁, hx₂⟩
    -/
    exact ⟨x₁, rfl⟩
    /-
      🎉 no goals
    -/


lemma ab_exact_iff_function_exact :
    S.Exact ↔ Function.Exact S.f S.g := by
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff S.Exact (Function.Exact ⇑S.f ⇑S.g)
  -/
  rw [S.ab_exact_iff]
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff (∀ (x₂ : ↑S.X₂), Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂) (Funct …
  -/
  apply forall_congr'
  /-
    case h
    S : CategoryTheory.ShortComplex Ab
    ⊢ ∀ (a : ↑S.X₂), Iff (Eq (S.g a) 0 → Exists fun x₁ => Eq (S.f x₁) a) (Iff (Eq  …
  -/
  intro x₂
  /-
    case h
    S : CategoryTheory.ShortComplex Ab
    x₂ : ↑S.X₂
    ⊢ Iff (Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂) (Iff (Eq (S.g x₂) 0) ( …
  -/
  constructor
    /-
      case h.mp
      S : CategoryTheory.ShortComplex Ab
      x₂ : ↑S.X₂
      ⊢ (Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂) → Iff (Eq (S.g x₂) 0) (Mem …
    -/
  · intro h
    /-
      case h.mp
      S : CategoryTheory.ShortComplex Ab
      x₂ : ↑S.X₂
      h : Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂
      ⊢ Iff (Eq (S.g x₂) 0) (Membership.mem (Set.range ⇑S.f) x₂)
    -/
    refine ⟨h, ?_⟩
    /-
      case h.mp
      S : CategoryTheory.ShortComplex Ab
      x₂ : ↑S.X₂
      h : Eq (S.g x₂) 0 → Exists fun x₁ => Eq (S.f x₁) x₂
      ⊢ Membership.mem (Set.range ⇑S.f) x₂ → Eq (S.g x₂) 0
    -/
    rintro ⟨x₁, rfl⟩
    /-
      case h.mp.intro
      S : CategoryTheory.ShortComplex Ab
      x₁ : ↑S.X₁
      h : Eq (S.g (S.f x₁)) 0 → Exists fun x₁_1 => Eq (S.f x₁_1) (S.f x₁)
      ⊢ Eq (S.g (S.f x₁)) 0
    -/
    simp only [ab_zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      S : CategoryTheory.ShortComplex Ab
      x₂ : ↑S.X₂
      ⊢ Iff (Eq (S.g x₂) 0) (Membership.mem (Set.range ⇑S.f) x₂) → Eq (S.g x₂) 0 → E …
    -/
  · tauto
    /-
      🎉 no goals
    -/


lemma ab_exact_iff_ker_le_range : S.Exact ↔ S.g.ker ≤ S.f.range := S.ab_exact_iff


lemma ab_exact_iff_range_eq_ker : S.Exact ↔ S.f.range = S.g.ker := by
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff S.Exact (Eq (AddMonoidHom.range S.f) (AddMonoidHom.ker S.g))
  -/
  rw [ab_exact_iff_ker_le_range]
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff (LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f)) (Eq (AddMonoidHo …
  -/
  constructor
    /-
      case mp
      S : CategoryTheory.ShortComplex Ab
      ⊢ LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f) → Eq (AddMonoidHom.ran …
    -/
  · intro h
    /-
      case mp
      S : CategoryTheory.ShortComplex Ab
      h : LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f)
      ⊢ Eq (AddMonoidHom.range S.f) (AddMonoidHom.ker S.g)
    -/
    refine le_antisymm ?_ h
    /-
      case mp
      S : CategoryTheory.ShortComplex Ab
      h : LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f)
      ⊢ LE.le (AddMonoidHom.range S.f) (AddMonoidHom.ker S.g)
    -/
    rintro _ ⟨x₁, rfl⟩
    /-
      case mp.intro
      S : CategoryTheory.ShortComplex Ab
      h : LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f)
      x₁ : ↑S.X₁
      ⊢ Membership.mem (AddMonoidHom.ker S.g) (S.f x₁)
    -/
    rw [AddMonoidHom.mem_ker, ← comp_apply, S.zero]
    /-
      case mp.intro
      S : CategoryTheory.ShortComplex Ab
      h : LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f)
      x₁ : ↑S.X₁
      ⊢ Eq (0 x₁) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S : CategoryTheory.ShortComplex Ab
      ⊢ Eq (AddMonoidHom.range S.f) (AddMonoidHom.ker S.g) → LE.le (AddMonoidHom.ker …
    -/
  · intro h
    /-
      case mpr
      S : CategoryTheory.ShortComplex Ab
      h : Eq (AddMonoidHom.range S.f) (AddMonoidHom.ker S.g)
      ⊢ LE.le (AddMonoidHom.ker S.g) (AddMonoidHom.range S.f)
    -/
    rw [h]
    /-
      🎉 no goals
    -/


lemma ShortExact.ab_injective_f (hS : S.ShortExact) :
    Function.Injective S.f :=
  (AddCommGrp.mono_iff_injective _).1 hS.mono_f


lemma ShortExact.ab_surjective_g (hS : S.ShortExact) :
    Function.Surjective S.g :=
  (AddCommGrp.epi_iff_surjective _).1 hS.epi_g


lemma ShortExact.ab_exact_iff_function_exact :
    S.Exact ↔ Function.Exact S.f S.g := by
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff S.Exact (Function.Exact ⇑S.f ⇑S.g)
  -/
  rw [ab_exact_iff_range_eq_ker, AddMonoidHom.exact_iff]
  /-
    S : CategoryTheory.ShortComplex Ab
    ⊢ Iff (Eq (AddMonoidHom.range S.f) (AddMonoidHom.ker S.g)) (Eq (AddMonoidHom.k …
  -/
  tauto
  /-
    🎉 no goals
  -/


