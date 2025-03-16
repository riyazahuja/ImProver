/-- A totally separated space is T2. -/
instance TotallySeparatedSpace.t2Space [TotallySeparatedSpace X] : T2Space X where
  t2 x y h := by
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TotallySeparatedSpace X
      x y : X
      h : Ne x y
      ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
    -/
    obtain ⟨u, v, h₁, h₂, h₃, h₄, _, h₅⟩ := isTotallySeparated_univ trivial trivial h
    /-
      case intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TotallySeparatedSpace X
      x y : X
      h : Ne x y
      u v : Set X
      h₁ : IsOpen u
      h₂ : IsOpen v
      h₃ : Membership.mem u x
      h₄ : Membership.mem v y
      left✝ : HasSubset.Subset Set.univ (Union.union u v)
      h₅ : Disjoint u v
      ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
    -/
    exact ⟨u, v, h₁, h₂, h₃, h₄, h₅⟩
    /-
      🎉 no goals
    -/


