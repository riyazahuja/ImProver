open Matrix in
instance matrix (ι : Type*) [Fintype ι] [DecidableEq ι] :
    Algebra.IsCentral K (Matrix ι ι D) where
  out m h := by
    refine isEmpty_or_nonempty ι |>.recOn
      (fun h => Algebra.mem_bot.2 ⟨0, Matrix.ext fun i _ => h.elim i⟩) fun ⟨i⟩ => ?_
    obtain ⟨d, rfl⟩ := mem_range_scalar_of_commute_stdBasisMatrix (M := m) (fun _ _ _ =>
      Subalgebra.mem_center_iff.mp h _)
    have mem : d ∈ Subalgebra.center K D := by
      rw [Subalgebra.mem_center_iff] at h ⊢
      intro d'
      simpa using Matrix.ext_iff.2 (h (scalar ι d')) i i
    /-
      case intro
      K : Type u_1
      D : Type u_2
      inst✝⁵ : CommSemiring K
      inst✝⁴ : Semiring D
      inst✝³ : Algebra K D
      inst✝² : Algebra.IsCentral K D
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x✝ : Nonempty ι
      i : ι
      d : D
      h : Membership.mem (Subalgebra.center K (Matrix ι ι D)) ((Matrix.scalar ι) d)
      mem : Membership.mem (Subalgebra.center K D) d
      ⊢ Membership.mem Bot.bot ((Matrix.scalar ι) d)
    -/
    rw [center_eq_bot, Algebra.mem_bot] at mem
    /-
      case intro
      K : Type u_1
      D : Type u_2
      inst✝⁵ : CommSemiring K
      inst✝⁴ : Semiring D
      inst✝³ : Algebra K D
      inst✝² : Algebra.IsCentral K D
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x✝ : Nonempty ι
      i : ι
      d : D
      h : Membership.mem (Subalgebra.center K (Matrix ι ι D)) ((Matrix.scalar ι) d)
      mem : Membership.mem (Set.range ⇑(algebraMap K D)) d
      ⊢ Membership.mem Bot.bot ((Matrix.scalar ι) d)
    -/
    obtain ⟨r, rfl⟩ := mem
    /-
      case intro.intro
      K : Type u_1
      D : Type u_2
      inst✝⁵ : CommSemiring K
      inst✝⁴ : Semiring D
      inst✝³ : Algebra K D
      inst✝² : Algebra.IsCentral K D
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x✝ : Nonempty ι
      i : ι
      r : K
      h : Membership.mem (Subalgebra.center K (Matrix ι ι D)) ((Matrix.scalar ι) ((a …
      ⊢ Membership.mem Bot.bot ((Matrix.scalar ι) ((algebraMap K D) r))
    -/
    rw [Algebra.mem_bot]
    /-
      case intro.intro
      K : Type u_1
      D : Type u_2
      inst✝⁵ : CommSemiring K
      inst✝⁴ : Semiring D
      inst✝³ : Algebra K D
      inst✝² : Algebra.IsCentral K D
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      x✝ : Nonempty ι
      i : ι
      r : K
      h : Membership.mem (Subalgebra.center K (Matrix ι ι D)) ((Matrix.scalar ι) ((a …
      ⊢ Membership.mem (Set.range ⇑(algebraMap K (Matrix ι ι D))) ((Matrix.scalar ι) …
    -/
    exact ⟨r, rfl⟩
    /-
      🎉 no goals
    -/


