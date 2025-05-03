/-- A square consisting of opens `X₂ ⊓ X₃`, `X₂`, `X₃` and `X₂ ⊔ X₃` is
a Mayer-Vietoris square. -/
@[simps! toSquare]
noncomputable def mayerVietorisSquare' (sq : Square (Opens T))
    (h₄ : sq.X₄ = sq.X₂ ⊔ sq.X₃) (h₁ : sq.X₁ = sq.X₂ ⊓ sq.X₃) :
    (Opens.grothendieckTopology T).MayerVietorisSquare :=
  GrothendieckTopology.MayerVietorisSquare.mk_of_isPullback
    (J := (Opens.grothendieckTopology T)) sq
    (Square.IsPullback.mk _ (by
      /-
        T : Type u
        inst✝ : TopologicalSpace T
        sq : CategoryTheory.Square (TopologicalSpace.Opens T)
        h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
        h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
        ⊢ CategoryTheory.Limits.IsLimit sq.pullbackCone
      -/
      refine PullbackCone.IsLimit.mk _ ?_ ?_ ?_ ?_
        /-
          case refine_1
          T : Type u
          inst✝ : TopologicalSpace T
          sq : CategoryTheory.Square (TopologicalSpace.Opens T)
          h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
          h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
          ⊢ (s : CategoryTheory.Limits.PullbackCone sq.f₂₄ sq.f₃₄) → Quiver.Hom s.pt sq.X₁
        -/
      · intro s
        /-
          case refine_1
          T : Type u
          inst✝ : TopologicalSpace T
          sq : CategoryTheory.Square (TopologicalSpace.Opens T)
          h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
          h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
          s : CategoryTheory.Limits.PullbackCone sq.f₂₄ sq.f₃₄
          ⊢ Quiver.Hom s.pt sq.X₁
        -/
        apply homOfLE
        /-
          case refine_1.h
          T : Type u
          inst✝ : TopologicalSpace T
          sq : CategoryTheory.Square (TopologicalSpace.Opens T)
          h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
          h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
          s : CategoryTheory.Limits.PullbackCone sq.f₂₄ sq.f₃₄
          ⊢ LE.le s.pt sq.X₁
        -/
        rw [h₁, le_inf_iff]
        /-
          case refine_1.h
          T : Type u
          inst✝ : TopologicalSpace T
          sq : CategoryTheory.Square (TopologicalSpace.Opens T)
          h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
          h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
          s : CategoryTheory.Limits.PullbackCone sq.f₂₄ sq.f₃₄
          ⊢ And (LE.le s.pt sq.X₂) (LE.le s.pt sq.X₃)
        -/
        exact ⟨leOfHom s.fst, leOfHom s.snd⟩
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        T : Type u
        inst✝ : TopologicalSpace T
        sq : CategoryTheory.Square (TopologicalSpace.Opens T)
        h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
        h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
        ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone sq.f₂₄ sq.f₃₄), Eq (CategoryTheory …
      -/
      all_goals intros; apply Subsingleton.elim))
      /-
        🎉 no goals
      -/
    (fun x hx ↦ by
      /-
        T : Type u
        inst✝ : TopologicalSpace T
        sq : CategoryTheory.Square (TopologicalSpace.Opens T)
        h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
        h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
        x : T
        hx : Membership.mem sq.X₄ x
        ⊢ Exists fun U => Exists fun f => And ((CategoryTheory.Sieve.ofTwoArrows sq.f₂ …
      -/
      rw [h₄] at hx
      /-
        T : Type u
        inst✝ : TopologicalSpace T
        sq : CategoryTheory.Square (TopologicalSpace.Opens T)
        h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
        h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
        x : T
        hx : Membership.mem (Max.max sq.X₂ sq.X₃) x
        ⊢ Exists fun U => Exists fun f => And ((CategoryTheory.Sieve.ofTwoArrows sq.f₂ …
      -/
      obtain (hx|hx) := hx
        /-
          case inl
          T : Type u
          inst✝ : TopologicalSpace T
          sq : CategoryTheory.Square (TopologicalSpace.Opens T)
          h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
          h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
          x : T
          hx : Membership.mem (↑sq.X₂) x
          ⊢ Exists fun U => Exists fun f => And ((CategoryTheory.Sieve.ofTwoArrows sq.f₂ …
        -/
      · exact ⟨_, _, ⟨Sieve.ofArrows_mk _ _ WalkingPair.left, hx⟩⟩
        /-
          🎉 no goals
        -/
        /-
          case inr
          T : Type u
          inst✝ : TopologicalSpace T
          sq : CategoryTheory.Square (TopologicalSpace.Opens T)
          h₄ : Eq sq.X₄ (Max.max sq.X₂ sq.X₃)
          h₁ : Eq sq.X₁ (Min.min sq.X₂ sq.X₃)
          x : T
          hx : Membership.mem (↑sq.X₃) x
          ⊢ Exists fun U => Exists fun f => And ((CategoryTheory.Sieve.ofTwoArrows sq.f₂ …
        -/
      · exact ⟨_, _, ⟨Sieve.ofArrows_mk _ _ WalkingPair.right, hx⟩⟩)
        /-
          🎉 no goals
        -/


/-- The Mayer-Vietoris square attached to two open subsets
of a topological space. -/
@[simps!]
noncomputable def mayerVietorisSquare (U V : Opens T):
    (Opens.grothendieckTopology T).MayerVietorisSquare :=
  mayerVietorisSquare'
    { X₁ := U ⊓ V
      X₂ := U
      X₃ := V
      X₄ := U ⊔ V
      f₁₂ := homOfLE inf_le_left
      f₁₃ := homOfLE inf_le_right
      f₂₄ := homOfLE le_sup_left
      f₃₄ := homOfLE le_sup_right
      fac := Subsingleton.elim _ _ } rfl rfl


