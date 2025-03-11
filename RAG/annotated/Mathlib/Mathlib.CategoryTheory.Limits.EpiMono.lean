lemma mono_iff_fst_eq_snd (hc : IsLimit c) : Mono f ↔ c.fst = c.snd := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.PullbackCone f f
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Iff (CategoryTheory.Mono f) (Eq c.fst c.snd)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Mono f → Eq c.fst c.snd
    -/
  · intro hf
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      hf : CategoryTheory.Mono f
      ⊢ Eq c.fst c.snd
    -/
    simpa only [← cancel_mono f] using c.condition
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Eq c.fst c.snd → CategoryTheory.Mono f
    -/
  · intro hf
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      hf : Eq c.fst c.snd
      ⊢ CategoryTheory.Mono f
    -/
    constructor
    /-
      case mpr.right_cancellation
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      hf : Eq c.fst c.snd
      ⊢ ∀ {Z : C} (g h : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp g f …
    -/
    intro Z g g' h
    /-
      case mpr.right_cancellation
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      hf : Eq c.fst c.snd
      Z : C
      g g' : Quiver.Hom Z X
      h : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruct …
      ⊢ Eq g g'
    -/
    obtain ⟨φ, rfl, rfl⟩ := PullbackCone.IsLimit.lift' hc g g' h
    /-
      case mpr.right_cancellation.mk.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      hf : Eq c.fst c.snd
      Z : C
      φ : Quiver.Hom Z c.pt
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.CategoryStru …
    -/
    rw [hf]
    /-
      🎉 no goals
    -/


lemma mono_iff_isIso_fst (hc : IsLimit c) : Mono f ↔ IsIso c.fst := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.PullbackCone f f
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Iff (CategoryTheory.Mono f) (CategoryTheory.IsIso c.fst)
  -/
  rw [mono_iff_fst_eq_snd hc]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.PullbackCone f f
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Iff (Eq c.fst c.snd) (CategoryTheory.IsIso c.fst)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Eq c.fst c.snd → CategoryTheory.IsIso c.fst
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      h : Eq c.fst c.snd
      ⊢ CategoryTheory.IsIso c.fst
    -/
    obtain ⟨φ, hφ₁, hφ₂⟩ := PullbackCone.IsLimit.lift' hc (𝟙 X) (𝟙 X) (by simp)
    /-
      case mp.mk.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      h : Eq c.fst c.snd
      φ : Quiver.Hom X c.pt
      hφ₁ : Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.Category …
      hφ₂ : Eq (CategoryTheory.CategoryStruct.comp φ c.snd) (CategoryTheory.Category …
      ⊢ CategoryTheory.IsIso c.fst
    -/
    refine ⟨φ, PullbackCone.IsLimit.hom_ext hc ?_ ?_, hφ₁⟩
      /-
        case mp.mk.intro.refine_1
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PullbackCone f f
        hc : CategoryTheory.Limits.IsLimit c
        h : Eq c.fst c.snd
        φ : Quiver.Hom X c.pt
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp φ c.snd) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp c …
      -/
    · dsimp
      /-
        case mp.mk.intro.refine_1
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PullbackCone f f
        hc : CategoryTheory.Limits.IsLimit c
        h : Eq c.fst c.snd
        φ : Quiver.Hom X c.pt
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp φ c.snd) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp c …
      -/
      simp only [assoc, hφ₁, id_comp, comp_id]
      /-
        🎉 no goals
      -/
      /-
        case mp.mk.intro.refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PullbackCone f f
        hc : CategoryTheory.Limits.IsLimit c
        h : Eq c.fst c.snd
        φ : Quiver.Hom X c.pt
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp φ c.snd) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp c …
      -/
    · dsimp
      /-
        case mp.mk.intro.refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PullbackCone f f
        hc : CategoryTheory.Limits.IsLimit c
        h : Eq c.fst c.snd
        φ : Quiver.Hom X c.pt
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp φ c.snd) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp c …
      -/
      simp only [assoc, hφ₂, id_comp, comp_id, h]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.IsIso c.fst → Eq c.fst c.snd
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      a✝ : CategoryTheory.IsIso c.fst
      ⊢ Eq c.fst c.snd
    -/
    obtain ⟨φ, hφ₁, hφ₂⟩ := PullbackCone.IsLimit.lift' hc (𝟙 X) (𝟙 X) (by simp)
    have : IsSplitEpi φ := IsSplitEpi.mk ⟨SplitEpi.mk c.fst (by
      rw [← cancel_mono c.fst, assoc, id_comp, hφ₁, comp_id])⟩
    /-
      case mpr.mk.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PullbackCone f f
      hc : CategoryTheory.Limits.IsLimit c
      a✝ : CategoryTheory.IsIso c.fst
      φ : Quiver.Hom X c.pt
      hφ₁ : Eq (CategoryTheory.CategoryStruct.comp φ c.fst) (CategoryTheory.Category …
      hφ₂ : Eq (CategoryTheory.CategoryStruct.comp φ c.snd) (CategoryTheory.Category …
      this : CategoryTheory.IsSplitEpi φ
      ⊢ Eq c.fst c.snd
    -/
    rw [← cancel_epi φ, hφ₁, hφ₂]
    /-
      🎉 no goals
    -/


lemma mono_iff_isIso_snd (hc : IsLimit c) : Mono f ↔ IsIso c.snd :=
  mono_iff_isIso_fst (PullbackCone.flipIsLimit hc)


lemma mono_iff_isPullback : Mono f ↔ IsPullback (𝟙 X) (𝟙 X) f f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (CategoryTheory.IsPullback (CategoryTheory.Categ …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono f → CategoryTheory.IsPullback (CategoryTheory.CategorySt …
    -/
  · intro
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id X) (CategoryTheo …
    -/
    exact IsPullback.of_isLimit (PullbackCone.isLimitMkIdId f)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id X) (CategoryTheo …
    -/
  · intro hf
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.id X) (CategoryT …
      ⊢ CategoryTheory.Mono f
    -/
    exact (mono_iff_fst_eq_snd hf.isLimit).2 rfl
    /-
      🎉 no goals
    -/


lemma epi_iff_inl_eq_inr (hc : IsColimit c) : Epi f ↔ c.inl = c.inr := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.PushoutCocone f f
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ Iff (CategoryTheory.Epi f) (Eq c.inl c.inr)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Epi f → Eq c.inl c.inr
    -/
  · intro hf
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      hf : CategoryTheory.Epi f
      ⊢ Eq c.inl c.inr
    -/
    simpa only [← cancel_epi f] using c.condition
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ Eq c.inl c.inr → CategoryTheory.Epi f
    -/
  · intro hf
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      hf : Eq c.inl c.inr
      ⊢ CategoryTheory.Epi f
    -/
    constructor
    /-
      case mpr.left_cancellation
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      hf : Eq c.inl c.inr
      ⊢ ∀ {Z : C} (g h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp f g …
    -/
    intro Z g g' h
    /-
      case mpr.left_cancellation
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      hf : Eq c.inl c.inr
      Z : C
      g g' : Quiver.Hom Y Z
      h : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
      ⊢ Eq g g'
    -/
    obtain ⟨φ, rfl, rfl⟩ := PushoutCocone.IsColimit.desc' hc g g' h
    /-
      case mpr.left_cancellation.mk.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      hf : Eq c.inl c.inr
      Z : C
      φ : Quiver.Hom c.pt Z
      h : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.CategoryStru …
    -/
    rw [hf]
    /-
      🎉 no goals
    -/


lemma epi_iff_isIso_inl (hc : IsColimit c) : Epi f ↔ IsIso c.inl := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.PushoutCocone f f
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ Iff (CategoryTheory.Epi f) (CategoryTheory.IsIso c.inl)
  -/
  rw [epi_iff_inl_eq_inr hc]
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.PushoutCocone f f
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ Iff (Eq c.inl c.inr) (CategoryTheory.IsIso c.inl)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ Eq c.inl c.inr → CategoryTheory.IsIso c.inl
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      h : Eq c.inl c.inr
      ⊢ CategoryTheory.IsIso c.inl
    -/
    obtain ⟨φ, hφ₁, hφ₂⟩ := PushoutCocone.IsColimit.desc' hc (𝟙 Y) (𝟙 Y) (by simp)
    /-
      case mp.mk.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      h : Eq c.inl c.inr
      φ : Quiver.Hom c.pt Y
      hφ₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.Category …
      hφ₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr φ) (CategoryTheory.Category …
      ⊢ CategoryTheory.IsIso c.inl
    -/
    refine ⟨φ, hφ₁, PushoutCocone.IsColimit.hom_ext hc ?_ ?_⟩
      /-
        case mp.mk.intro.refine_1
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PushoutCocone f f
        hc : CategoryTheory.Limits.IsColimit c
        h : Eq c.inl c.inr
        φ : Quiver.Hom c.pt Y
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr φ) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
      -/
    · dsimp
      /-
        case mp.mk.intro.refine_1
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PushoutCocone f f
        hc : CategoryTheory.Limits.IsColimit c
        h : Eq c.inl c.inr
        φ : Quiver.Hom c.pt Y
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr φ) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
      -/
      simp only [comp_id, reassoc_of% hφ₁]
      /-
        🎉 no goals
      -/
      /-
        case mp.mk.intro.refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PushoutCocone f f
        hc : CategoryTheory.Limits.IsColimit c
        h : Eq c.inl c.inr
        φ : Quiver.Hom c.pt Y
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr φ) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr (CategoryTheory.CategoryStruct. …
      -/
    · dsimp
      /-
        case mp.mk.intro.refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.PushoutCocone f f
        hc : CategoryTheory.Limits.IsColimit c
        h : Eq c.inl c.inr
        φ : Quiver.Hom c.pt Y
        hφ₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.Category …
        hφ₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr φ) (CategoryTheory.Category …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr (CategoryTheory.CategoryStruct. …
      -/
      simp only [comp_id, h, reassoc_of% hφ₂]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.IsIso c.inl → Eq c.inl c.inr
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      a✝ : CategoryTheory.IsIso c.inl
      ⊢ Eq c.inl c.inr
    -/
    obtain ⟨φ, hφ₁, hφ₂⟩ := PushoutCocone.IsColimit.desc' hc (𝟙 Y) (𝟙 Y) (by simp)
    have : IsSplitMono φ := IsSplitMono.mk ⟨SplitMono.mk c.inl (by
      rw [← cancel_epi c.inl, reassoc_of% hφ₁, comp_id])⟩
    /-
      case mpr.mk.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      c : CategoryTheory.Limits.PushoutCocone f f
      hc : CategoryTheory.Limits.IsColimit c
      a✝ : CategoryTheory.IsIso c.inl
      φ : Quiver.Hom c.pt Y
      hφ₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl φ) (CategoryTheory.Category …
      hφ₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr φ) (CategoryTheory.Category …
      this : CategoryTheory.IsSplitMono φ
      ⊢ Eq c.inl c.inr
    -/
    rw [← cancel_mono φ, hφ₁, hφ₂]
    /-
      🎉 no goals
    -/


lemma epi_iff_isIso_inr (hc : IsColimit c) : Epi f ↔ IsIso c.inr :=
  epi_iff_isIso_inl (PushoutCocone.flipIsColimit hc)


lemma epi_iff_isPushout : Epi f ↔ IsPushout f f (𝟙 Y) (𝟙 Y) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (CategoryTheory.IsPushout f f (CategoryTheory.Cat …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → CategoryTheory.IsPushout f f (CategoryTheory.Category …
    -/
  · intro
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsPushout f f (CategoryTheory.CategoryStruct.id Y) (CategoryT …
    -/
    exact IsPushout.of_isColimit (PushoutCocone.isColimitMkIdId f)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.IsPushout f f (CategoryTheory.CategoryStruct.id Y) (CategoryT …
    -/
  · intro hf
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.IsPushout f f (CategoryTheory.CategoryStruct.id Y) (Catego …
      ⊢ CategoryTheory.Epi f
    -/
    exact (epi_iff_inl_eq_inr hf.isColimit).2 rfl
    /-
      🎉 no goals
    -/


