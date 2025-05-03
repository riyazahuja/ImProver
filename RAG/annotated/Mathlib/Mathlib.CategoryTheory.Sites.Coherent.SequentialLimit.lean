private structure struct (F : ℕᵒᵖ ⥤ Sheaf (coherentTopology C) (Type v)) where
  X (n : ℕ) : C
  x (n : ℕ) : (F.obj ⟨n⟩).val.obj ⟨X n⟩
  map (n : ℕ) : X (n + 1) ⟶ X n
  effectiveEpi (n : ℕ) : EffectiveEpi (map n)
  w (n : ℕ) : (F.map (homOfLE (n.le_add_right 1)).op).val.app (op (X (n + 1))) (x (n + 1)) =
      (F.obj (op n)).val.map (map n).op (x n)


include hF in
private lemma exists_effectiveEpi (n : ℕ) (X : C) (y : (F.obj ⟨n⟩).val.obj ⟨X⟩) :
    ∃ (X' : C) (φ : X' ⟶ X) (_ : EffectiveEpi φ) (x : (F.obj ⟨n + 1⟩).val.obj ⟨X'⟩),
      ((F.map (homOfLE (n.le_add_right 1)).op).val.app ⟨X'⟩) x = ((F.obj ⟨n⟩).val.map φ.op) y := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    n : Nat
    X : C
    y : (F.obj { unop := n }).val.obj { unop := X }
    ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((F.map  …
  -/
  have := hF n
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    n : Nat
    X : C
    y : (F.obj { unop := n }).val.obj { unop := X }
    this : CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryTheory.homOfLE …
    ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((F.map  …
  -/
  rw [coherentTopology.isLocallySurjective_iff, regularTopology.isLocallySurjective_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    n : Nat
    X : C
    y : (F.obj { unop := n }).val.obj { unop := X }
    this : ∀ (X : C) (y : (CategoryTheory.forget (Type v)).obj ((F.obj { unop := n …
    ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((F.map  …
  -/
  exact this X y
  /-
    🎉 no goals
  -/


private noncomputable def preimage (X : C) (y : (F.obj ⟨0⟩).val.obj ⟨X⟩) :
    (n : ℕ) → ((Y : C) × (F.obj ⟨n⟩).val.obj ⟨Y⟩)
  | 0 => ⟨X, y⟩
  | (n+1) => ⟨(exists_effectiveEpi hF n (preimage X y n).1 (preimage X y n).2).choose,
      (exists_effectiveEpi hF n
        (preimage X y n).1 (preimage X y n).2).choose_spec.choose_spec.choose_spec.choose⟩


private noncomputable def preimageStruct (X : C) (y : (F.obj ⟨0⟩).val.obj ⟨X⟩) : struct F where
  X n := (preimage hF X y n).1
  x n := (preimage hF X y n).2
  map n := (exists_effectiveEpi hF n _ _).choose_spec.choose
  effectiveEpi n := (exists_effectiveEpi hF n _ _).choose_spec.choose_spec.choose
  w n := (exists_effectiveEpi hF n _ _).choose_spec.choose_spec.choose_spec.choose_spec


private noncomputable def preimageDiagram (X : C) (y : (F.obj ⟨0⟩).val.obj ⟨X⟩) : ℕᵒᵖ ⥤ C :=
  Functor.ofOpSequence (preimageStruct hF X y).map


private noncomputable def cone (X : C) (y : (F.obj ⟨0⟩).val.obj ⟨X⟩) : Cone F where
  pt := ((coherentTopology C).yoneda).obj (limit (preimageDiagram hF X y))
  π := NatTrans.ofOpSequence
    (fun n ↦ (coherentTopology C).yoneda.map
      (limit.π _ ⟨n⟩) ≫ ((coherentTopology C).yonedaEquiv).symm ((preimageStruct hF X y).x n)) (by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preregular C
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite Nat) C
      X : C
      y : (F.obj { unop := 0 }).val.obj { unop := X }
      ⊢ ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functo …
    -/
    intro n
    simp only [Functor.const_obj_obj, homOfLE_leOfHom, Functor.const_obj_map, Category.id_comp,
      Category.assoc, ← limit.w (preimageDiagram hF X y) (homOfLE (n.le_add_right 1)).op,
      homOfLE_leOfHom, Functor.map_comp]
    simp [GrothendieckTopology.yonedaEquiv_symm_naturality_left,
      GrothendieckTopology.yonedaEquiv_symm_naturality_right,
      preimageDiagram, (preimageStruct hF X y).w n])


include hF h hc in
lemma isLocallySurjective_π_app_zero_of_isLocallySurjective_map  :
    Sheaf.IsLocallySurjective (c.π.app ⟨0⟩) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite Nat) C
    h : ∀ (G : CategoryTheory.Functor (Opposite Nat) C), (∀ (n : Nat), CategoryThe …
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective (c.π.app { unop := 0 })
  -/
  rw [coherentTopology.isLocallySurjective_iff, regularTopology.isLocallySurjective_iff]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite Nat) C
    h : ∀ (G : CategoryTheory.Functor (Opposite Nat) C), (∀ (n : Nat), CategoryThe …
    ⊢ ∀ (X : C) (y : (CategoryTheory.forget (Type v)).obj ((F.obj { unop := 0 }).v …
  -/
  intro X y
  have hh : EffectiveEpi (limit.π (preimageDiagram hF X y) ⟨0⟩) :=
    h _ fun n ↦ by simpa [preimageDiagram] using (preimageStruct hF X y).effectiveEpi n
  refine ⟨limit (preimageDiagram hF X y), limit.π (preimageDiagram hF X y) ⟨0⟩, hh,
    (coherentTopology C).yonedaEquiv (hc.lift (cone hF X y )),
    (?_ : (c.π.app (op 0)).val.app _ _ = _)⟩
  simp only [← (coherentTopology C).yonedaEquiv_comp, Functor.const_obj_obj, cone,
    IsLimit.fac, NatTrans.ofOpSequence_app, (coherentTopology C).yonedaEquiv_comp,
    (coherentTopology C).yonedaEquiv_yoneda_map]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite Nat) C
    h : ∀ (G : CategoryTheory.Functor (Opposite Nat) C), (∀ (n : Nat), CategoryThe …
    X : C
    y : (CategoryTheory.forget (Type v)).obj ((F.obj { unop := 0 }).val.obj { unop …
    hh : CategoryTheory.EffectiveEpi (CategoryTheory.Limits.limit.π (CategoryTheor …
    ⊢ Eq (((CategoryTheory.coherentTopology C).yonedaEquiv.symm ((CategoryTheory.c …
  -/
  rfl
  /-
    🎉 no goals
  -/


include h in
lemma epi_π_app_zero_of_epi [HasSheafify (coherentTopology C) (Type v)]
    [Balanced (Sheaf (coherentTopology C) (Type v))]
    [(coherentTopology C).WEqualsLocallyBijective (Type v)]
    {F : ℕᵒᵖ ⥤ Sheaf (coherentTopology C) (Type v)}
    {c : Cone F} (hc : IsLimit c)
    (hF : ∀ n, Epi (F.map (homOfLE (Nat.le_succ n)).op)) : Epi (c.π.app ⟨0⟩) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Preregular C
    inst✝⁴ : CategoryTheory.FinitaryExtensive C
    inst✝³ : CategoryTheory.Limits.HasLimitsOfShape (Opposite Nat) C
    h : ∀ (G : CategoryTheory.Functor (Opposite Nat) C), (∀ (n : Nat), CategoryThe …
    inst✝² : CategoryTheory.HasSheafify (CategoryTheory.coherentTopology C) (Type v)
    inst✝¹ : CategoryTheory.Balanced (CategoryTheory.Sheaf (CategoryTheory.coheren …
    inst✝ : (CategoryTheory.coherentTopology C).WEqualsLocallyBijective (Type v)
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
    ⊢ CategoryTheory.Epi (c.π.app { unop := 0 })
  -/
  simp_rw [← Sheaf.isLocallySurjective_iff_epi'] at hF ⊢
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Preregular C
    inst✝⁴ : CategoryTheory.FinitaryExtensive C
    inst✝³ : CategoryTheory.Limits.HasLimitsOfShape (Opposite Nat) C
    h : ∀ (G : CategoryTheory.Functor (Opposite Nat) C), (∀ (n : Nat), CategoryThe …
    inst✝² : CategoryTheory.HasSheafify (CategoryTheory.coherentTopology C) (Type v)
    inst✝¹ : CategoryTheory.Balanced (CategoryTheory.Sheaf (CategoryTheory.coheren …
    inst✝ : (CategoryTheory.coherentTopology C).WEqualsLocallyBijective (Type v)
    F : CategoryTheory.Functor (Opposite Nat) (CategoryTheory.Sheaf (CategoryTheor …
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Sheaf.IsLocallySurjective (F.map (CategoryThe …
    ⊢ CategoryTheory.Sheaf.IsLocallySurjective (c.π.app { unop := 0 })
  -/
  exact isLocallySurjective_π_app_zero_of_isLocallySurjective_map hc hF h
  /-
    🎉 no goals
  -/


