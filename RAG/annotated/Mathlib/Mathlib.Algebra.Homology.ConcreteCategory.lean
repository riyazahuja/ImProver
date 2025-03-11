/-- Constructor for cycles of a homological complex in a concrete category. -/
noncomputable def cyclesMk {i : ι} (x : (forget₂ C Ab).obj (K.X i)) (j : ι) (hj : c.next i = j)
    (hx : ((forget₂ C Ab).map (K.d i j)) x = 0) :
    (forget₂ C Ab).obj (K.cycles i) :=
                          /-
                            C : Type u
                            inst✝⁵ : CategoryTheory.Category.{v, u} C
                            inst✝⁴ : CategoryTheory.ConcreteCategory C
                            inst✝³ : CategoryTheory.HasForget₂ C Ab
                            inst✝² : CategoryTheory.Abelian C
                            inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
                            inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
                            ι : Type u_1
                            c : ComplexShape ι
                            K : HomologicalComplex C c
                            i : ι
                            x : ↑((CategoryTheory.forget₂ C Ab).obj (K.X i))
                            j : ι
                            hj : Eq (c.next i) j
                            hx : Eq (((CategoryTheory.forget₂ C Ab).map (K.d i j)) x) 0
                            ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (K.sc i).g) x) 0
                          -/
  (K.sc i).cyclesMk x (by subst hj; exact hx)
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
lemma i_cyclesMk {i : ι} (x : (forget₂ C Ab).obj (K.X i)) (j : ι) (hj : c.next i = j)
    (hx : ((forget₂ C Ab).map (K.d i j)) x = 0) :
    ((forget₂ C Ab).map (K.iCycles i)) (K.cyclesMk x j hj hx) = x := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    ι : Type u_1
    c : ComplexShape ι
    K : HomologicalComplex C c
    i : ι
    x : ↑((CategoryTheory.forget₂ C Ab).obj (K.X i))
    j : ι
    hj : Eq (c.next i) j
    hx : Eq (((CategoryTheory.forget₂ C Ab).map (K.d i j)) x) 0
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (K.iCycles i)) (K.cyclesMk x j hj hx) …
  -/
  subst hj
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    ι : Type u_1
    c : ComplexShape ι
    K : HomologicalComplex C c
    i : ι
    x : ↑((CategoryTheory.forget₂ C Ab).obj (K.X i))
    hx : Eq (((CategoryTheory.forget₂ C Ab).map (K.d i (c.next i))) x) 0
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (K.iCycles i)) (K.cyclesMk x (c.next  …
  -/
  apply (K.sc i).i_cyclesMk
  /-
    🎉 no goals
  -/


lemma δ_apply' (x₃ : (forget₂ C Ab).obj (S.X₃.homology i))
    (x₂ : (forget₂ C Ab).obj (S.X₂.opcycles i))
    (x₁ : (forget₂ C Ab).obj (S.X₁.cycles j))
    (h₂ : (forget₂ C Ab).map (HomologicalComplex.opcyclesMap S.g i) x₂ =
      (forget₂ C Ab).map (S.X₃.homologyι i) x₃)
    (h₁ : (forget₂ C Ab).map (HomologicalComplex.cyclesMap S.f j) x₁ =
      (forget₂ C Ab).map (S.X₂.opcyclesToCycles i j) x₂) :
    (forget₂ C Ab).map (hS.δ i j hij) x₃ = (forget₂ C Ab).map (S.X₁.homologyπ j) x₁ :=
  (HomologicalComplex.HomologySequence.snakeInput hS i j hij).δ_apply' x₃ x₂ x₁ h₂ h₁


lemma δ_apply (x₃ : (forget₂ C Ab).obj (S.X₃.X i))
    (hx₃ : (forget₂ C Ab).map (S.X₃.d i j) x₃ = 0)
    (x₂ : (forget₂ C Ab).obj (S.X₂.X i)) (hx₂ : (forget₂ C Ab).map (S.g.f i) x₂ = x₃)
    (x₁ : (forget₂ C Ab).obj (S.X₁.X j))
    (hx₁ : (forget₂ C Ab).map (S.f.f j) x₁ = (forget₂ C Ab).map (S.X₂.d i j) x₂)
    (k : ι) (hk : c.next j = k) :
    (forget₂ C Ab).map (hS.δ i j hij)
      ((forget₂ C Ab).map (S.X₃.homologyπ i) (S.X₃.cyclesMk x₃ j (c.next_eq' hij) hx₃)) =
        (forget₂ C Ab).map (S.X₁.homologyπ j) (S.X₁.cyclesMk x₁ k hk (by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.ConcreteCategory C
            inst✝³ : CategoryTheory.HasForget₂ C Ab
            inst✝² : CategoryTheory.Abelian C
            inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
            inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
            ι : Type u_1
            c : ComplexShape ι
            S : CategoryTheory.ShortComplex (HomologicalComplex C c)
            hS : S.ShortExact
            i j : ι
            hij : c.Rel i j
            x₃ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₃.X i))
            hx₃ : Eq (((CategoryTheory.forget₂ C Ab).map (S.X₃.d i j)) x₃) 0
            x₂ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₂.X i))
            hx₂ : Eq (((CategoryTheory.forget₂ C Ab).map (S.g.f i)) x₂) x₃
            x₁ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₁.X j))
            hx₁ : Eq (((CategoryTheory.forget₂ C Ab).map (S.f.f j)) x₁) (((CategoryTheory. …
            k : ι
            hk : Eq (c.next j) k
            ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (S.X₁.d j k)) x₁) 0
          -/
          have := hS.mono_f
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.ConcreteCategory C
            inst✝³ : CategoryTheory.HasForget₂ C Ab
            inst✝² : CategoryTheory.Abelian C
            inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
            inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
            ι : Type u_1
            c : ComplexShape ι
            S : CategoryTheory.ShortComplex (HomologicalComplex C c)
            hS : S.ShortExact
            i j : ι
            hij : c.Rel i j
            x₃ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₃.X i))
            hx₃ : Eq (((CategoryTheory.forget₂ C Ab).map (S.X₃.d i j)) x₃) 0
            x₂ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₂.X i))
            hx₂ : Eq (((CategoryTheory.forget₂ C Ab).map (S.g.f i)) x₂) x₃
            x₁ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₁.X j))
            hx₁ : Eq (((CategoryTheory.forget₂ C Ab).map (S.f.f j)) x₁) (((CategoryTheory. …
            k : ι
            hk : Eq (c.next j) k
            this : CategoryTheory.Mono S.f
            ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (S.X₁.d j k)) x₁) 0
          -/
          apply (Preadditive.mono_iff_injective (S.f.f k)).1 inferInstance
          rw [← forget₂_comp_apply, ← HomologicalComplex.Hom.comm, forget₂_comp_apply, hx₁,
            ← forget₂_comp_apply, HomologicalComplex.d_comp_d, Functor.map_zero, map_zero,
            AddMonoidHom.zero_apply])) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    ι : Type u_1
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.ShortExact
    i j : ι
    hij : c.Rel i j
    x₃ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₃.X i))
    hx₃ : Eq (((CategoryTheory.forget₂ C Ab).map (S.X₃.d i j)) x₃) 0
    x₂ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₂.X i))
    hx₂ : Eq (((CategoryTheory.forget₂ C Ab).map (S.g.f i)) x₂) x₃
    x₁ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₁.X j))
    hx₁ : Eq (((CategoryTheory.forget₂ C Ab).map (S.f.f j)) x₁) (((CategoryTheory. …
    k : ι
    hk : Eq (c.next j) k
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (hS.δ i j hij)) (((CategoryTheory.for …
  -/
  refine hS.δ_apply' i j hij _ ((forget₂ C Ab).map (S.X₂.pOpcycles i) x₂) _ ?_ ?_
  · rw [← forget₂_comp_apply, ← forget₂_comp_apply,
      HomologicalComplex.p_opcyclesMap, Functor.map_comp, comp_apply,
      HomologicalComplex.homology_π_ι, forget₂_comp_apply, hx₂, HomologicalComplex.i_cyclesMk]
    /-
      case refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      ι : Type u_1
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₃.X i))
      hx₃ : Eq (((CategoryTheory.forget₂ C Ab).map (S.X₃.d i j)) x₃) 0
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₂.X i))
      hx₂ : Eq (((CategoryTheory.forget₂ C Ab).map (S.g.f i)) x₂) x₃
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj (S.X₁.X j))
      hx₁ : Eq (((CategoryTheory.forget₂ C Ab).map (S.f.f j)) x₁) (((CategoryTheory. …
      k : ι
      hk : Eq (c.next j) k
      ⊢ Eq (((CategoryTheory.forget₂ C Ab).map (HomologicalComplex.cyclesMap S.f j)) …
    -/
  · apply (Preadditive.mono_iff_injective (S.X₂.iCycles j)).1 inferInstance
    conv_lhs =>
      rw [← forget₂_comp_apply, HomologicalComplex.cyclesMap_i, forget₂_comp_apply,
        HomologicalComplex.i_cyclesMk, hx₁]
    conv_rhs =>
      rw [← forget₂_comp_apply, ← forget₂_comp_apply,
        HomologicalComplex.pOpcycles_opcyclesToCycles_assoc, HomologicalComplex.toCycles_i]


