instance instUniformSpace : UniformSpace (Matrix m n 𝕜) :=
      /-
        m : Type u_1
        n : Type u_2
        𝕜 : Type u_3
        inst✝ : UniformSpace 𝕜
        ⊢ UniformSpace (m → n → 𝕜)
      -/
  (by infer_instance : UniformSpace (m → n → 𝕜))
      /-
        🎉 no goals
      -/


instance instUniformAddGroup [AddGroup 𝕜] [UniformAddGroup 𝕜] :
    UniformAddGroup (Matrix m n 𝕜) :=
  inferInstanceAs <| UniformAddGroup (m → n → 𝕜)


theorem uniformity :
    𝓤 (Matrix m n 𝕜) = ⨅ (i : m) (j : n), (𝓤 𝕜).comap fun a => (a.1 i j, a.2 i j) := by
  /-
    m : Type u_1
    n : Type u_2
    𝕜 : Type u_3
    inst✝ : UniformSpace 𝕜
    ⊢ Eq (_root_.uniformity (Matrix m n 𝕜)) (iInf fun i => iInf fun j => Filter.co …
  -/
  erw [Pi.uniformity]
  /-
    m : Type u_1
    n : Type u_2
    𝕜 : Type u_3
    inst✝ : UniformSpace 𝕜
    ⊢ Eq (iInf fun i => Filter.comap (fun a => { fst := a.1 i, snd := a.2 i }) (_r …
  -/
  simp_rw [Pi.uniformity, Filter.comap_iInf, Filter.comap_comap]
  /-
    m : Type u_1
    n : Type u_2
    𝕜 : Type u_3
    inst✝ : UniformSpace 𝕜
    ⊢ Eq (iInf fun i => iInf fun i_1 => Filter.comap (Function.comp (fun a => { fs …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem uniformContinuous {β : Type*} [UniformSpace β] {f : β → Matrix m n 𝕜} :
    UniformContinuous f ↔ ∀ i j, UniformContinuous fun x => f x i j := by
  /-
    m : Type u_1
    n : Type u_2
    𝕜 : Type u_3
    inst✝¹ : UniformSpace 𝕜
    β : Type u_4
    inst✝ : UniformSpace β
    f : β → Matrix m n 𝕜
    ⊢ Iff (UniformContinuous f) (∀ (i : m) (j : n), UniformContinuous fun x => f x …
  -/
  simp only [UniformContinuous, Matrix.uniformity, Filter.tendsto_iInf, Filter.tendsto_comap_iff]
  /-
    m : Type u_1
    n : Type u_2
    𝕜 : Type u_3
    inst✝¹ : UniformSpace 𝕜
    β : Type u_4
    inst✝ : UniformSpace β
    f : β → Matrix m n 𝕜
    ⊢ Iff (∀ (i : m) (i_1 : n), Filter.Tendsto (Function.comp (fun a => { fst := a …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  apply Iff.intro <;> intro a <;> apply a
                                  /-
                                    🎉 no goals
                                  -/


instance [CompleteSpace 𝕜] : CompleteSpace (Matrix m n 𝕜) :=
      /-
        m : Type u_1
        n : Type u_2
        𝕜 : Type u_3
        inst✝¹ : UniformSpace 𝕜
        inst✝ : CompleteSpace 𝕜
        ⊢ CompleteSpace (m → n → 𝕜)
      -/
  (by infer_instance : CompleteSpace (m → n → 𝕜))
      /-
        🎉 no goals
      -/


instance [T0Space 𝕜] : T0Space (Matrix m n 𝕜) :=
  inferInstanceAs (T0Space (m → n → 𝕜))


